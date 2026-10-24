"""身体化パーセプトロン（仮想筋骨格レイヤー）および比較ベースライン制御器モジュール

本モジュールは、四脚歩行ロボット（Unitree Go2相当、12自由度）の歩行制御器として以下の3つのモデルを定義する。
1. ProposedVM: 身体化パーセプトロン表現に基づく仮想筋骨格インピーダンス制御レイヤー＋極小神経制御器（提案手法）。
2. MinimalPD: 提案手法と同一の上位極小神経制御器＋関節独立線形PD制御器（アブレーション検証用）。
3. BaselineMLP: 3層多層パーセプトロン（MLP）＋関節独立線形PD制御器（従来手法ベースライン）。

数理的プレミス（前提条件）:
    - 物理的な筋腱複合体は引張力のみを発生・伝達し、圧縮方向にはたるみ（Slack）を生じて力を伝達しない。
      したがって、各仮想筋の張力 f_i は常に非負（f_i >= 0）でなければならない。
    - 解剖学的妥当性を保証するため、モーメントアーム行列 R (32 x 12) には各脚の8本の筋が当該脚の3関節のみを
      駆動するブロック対角構造制約（R_mask）を課し、異なる脚間を跨ぐ非物理的な交差結合を排除する。
    - 関節角度ベクトル q (12次元) から筋伸長変位 l (32次元) への幾何学的写像が l(q) = R q + b で与えられるとき、
      仮想仕事の原理（Principle of Virtual Work）により、筋張力 f が関節系に及ぼす一般化トルク tau は
      モーメントアーム行列 R の負転置を用いて tau = - R^T f と一意に定まる。
"""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """数値安定性を考慮したシグモイド活性化関数

    数式:
        sigma(x) = 1 / (1 + exp(-x))

    前提条件:
        指数関数のオーバーフローを防止するため、入力 x を [-15.0, 15.0] の範囲にクリップする。
        上位神経回路からのロジット入力を [0, 1] の正規化筋活動度へ変換するために用いる。

    Args:
        x (np.ndarray): 任意形状の実数入力配列（神経回路のロジット出力）。

    Returns:
        np.ndarray: (0, 1) の範囲に正規化された出力配列。
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


def relu(x: np.ndarray) -> np.ndarray:
    """腱のたるみ（Slack）非線形性を表現する正規化線形関数（ReLU）

    数式:
        phi(x) = max(0, x)

    前提条件:
        筋や腱は引張方向（x > 0）にのみ張力を発生し、圧縮方向（x <= 0）では座屈・たるみを生じて
        発生張力が厳密にゼロとなる物理的片側拘束（Unilateral constraint）をモデル化する。

    Args:
        x (np.ndarray): 各仮想筋の内部プレ張力配列。

    Returns:
        np.ndarray: 非負制約（>= 0）が適用された実効筋張力配列。
    """
    return np.maximum(0.0, x)


class ProposedVM:
    """提案手法: 仮想筋骨格レイヤー（身体化パーセプトロン）＋極小神経制御器

    32本の仮想筋（1脚あたり8本 × 4脚）と12個の関節（1脚あたり3関節 × 4脚）からなる二層構造の制御器。
    下位の仮想筋骨格レイヤーは3層パーセプトロンと数理的に等価なインピーダンス変換を行い、
    上位の極小神経制御器は位相信号と姿勢フィードバックから能動収縮指令を生成する。

    組み込まれている数理モデルと計算式:
        1. 上位神経制御器による能動収縮力（Active Contraction Force）の生成:
           f_active = F_max * sigma(W_neural @ x_osc + W_fb @ s_imu)
           ここで F_max = 24.0 [N] は最大能動収縮力、x_osc (4次元) はトロット位相基底、
           s_imu (6次元) は胴体姿勢および順運動学推定高さフィードバック。

        2. 仮想筋の伸長量（Strain）および伸長速度（Strain Rate）の幾何学的写像:
           strain = R @ q + b
           strain_rate = R @ dq
           ここで R (32 x 12) はブロック対角マスク R_mask により各脚内部に拘束されたモーメントアーム行列、
           b (32次元) は初期プリテンション変位。

        3. 腱のたるみ（ReLU）と粘弾性・ヒル型短縮アンローディング特性を含む非負筋張力の算出:
           f = max(0, K * (R @ q + b) + D * (R @ dq) + f_active)
           ここで K = exp(p_K) > 0 は仮想剛性、D = exp(p_D) > 0 は仮想粘性減衰係数。
           急速短縮時（R @ dq < 0）には粘性項が負となり張力を減衰させ、圧縮力への転化をReLUで遮断する。

        4. 仮想仕事の原理に基づく関節トルクへの変換:
           tau = - R^T @ f

    解剖学的筋配置（1脚あたり8本、j_s = 脚インデックス * 3）:
        - 筋0: 股関節外転筋（Hip Abductor, 関節0）
        - 筋1: 股関節内転筋（Hip Adductor, 関節0）
        - 筋2: 股関節屈筋（Hip Flexor, 関節1）
        - 筋3: 股関節伸筋（Hip Extensor, 関節1）
        - 筋4: 膝関節屈筋（Knee Flexor, 関節2）
        - 筋5: 膝関節伸筋（Knee Extensor, 関節2）
        - 筋6: 大腿直筋様二関節筋（Rectus Femoris-like, 股関節屈曲＋膝関節伸展）
        - 筋7: ハムストリングス様二関節筋（Hamstrings-like, 股関節伸展＋膝関節屈曲）

    Attributes:
        m (int): 仮想筋の総数（32本）。
        n (int): ロボットの駆動関節自由度数（12自由度）。
        R_mask (np.ndarray): 各脚内の筋・関節結合のみを許容する解剖学的ブロック対角マスク（32 x 12）。
        R (np.ndarray): 仮想モーメントアーム行列（形状: 32 x 12）。
        b (np.ndarray): 仮想筋の初期伸長バイアスベクトル（形状: 32）。
        p_K (np.ndarray): 仮想剛性の対数パラメータベクトル（形状: 32）。
        p_D (np.ndarray): 仮想減衰の対数パラメータベクトル（形状: 32）。
        W_neural (np.ndarray): 位相振動子入力に対する上位神経重み行列（形状: 32 x 4）。
        W_fb (np.ndarray): IMU姿勢入力に対するフィードバック重み行列（形状: 32 x 6）。
        F_max (float): 各仮想筋の最大能動収縮力 [N]。
    """

    def __init__(self, seed: int = 42):
        """ProposedVM の初期化と解剖学的初期パラメータの設定

        各脚の単関節筋および二関節筋のモーメントアーム符号、ブロック対角マスク R_mask、
        相反神経支配（Reciprocal Inhibition）を考慮した初期トロット歩容結合、
        および対数剛性・減衰パラメータを設定する。

        Args:
            seed (int): 乱数シード。
        """
        self.m = 32  # 仮想筋数: 1脚あたり8本 x 4脚 = 32本
        self.n = 12  # 関節数: 1脚あたり3関節 x 4脚 = 12関節

        # 解剖学的ブロック対角マスク R_mask の構築（異なる脚間の非物理的結合を厳密に遮断）
        self.R_mask = np.zeros((self.m, self.n), dtype=np.float64)
        for leg in range(4):
            self.R_mask[leg * 8 : (leg + 1) * 8, leg * 3 : (leg + 1) * 3] = 1.0

        # 1. 仮想モーメントアーム行列 R の初期化（32 x 12）
        # トルク変換式 tau = - R.T @ f （f >= 0）に基づく符号規約:
        # 正トルク（tau > 0）を発生させたい場合: -R.T > 0 より R = -1.0
        # 負トルク（tau < 0）を発生させたい場合: -R.T < 0 より R = +1.0
        R_init = np.zeros((self.m, self.n), dtype=np.float64)
        for leg in range(4):
            m_s = leg * 8  # 当該脚の筋インデックス開始位置
            j_s = leg * 3  # 当該脚の関節インデックス開始位置

            # 筋0: 股関節外転筋（脚を展開、tau > 0 -> R = -1.0）
            R_init[m_s + 0, j_s + 0] = -1.0
            # 筋1: 股関節内転筋（脚を引き込み、tau < 0 -> R = 1.0）
            R_init[m_s + 1, j_s + 0] = 1.0
            # 筋2: 股関節屈筋（大腿を前振出し、tau < 0 -> R = 1.0）
            R_init[m_s + 2, j_s + 1] = 1.0
            # 筋3: 股関節伸筋（大腿を後方キック、tau > 0 -> R = -1.0）
            R_init[m_s + 3, j_s + 1] = -1.0
            # 筋4: 膝関節屈筋（膝を曲げて足上げ、tau > 0 -> R = -1.0）
            R_init[m_s + 4, j_s + 2] = -1.0
            # 筋5: 膝関節伸筋（膝を伸ばして突っ張り、tau < 0 -> R = 1.0）
            R_init[m_s + 5, j_s + 2] = 1.0
            # 筋6: 二関節筋・大腿直筋様結合（大腿屈曲かつ膝伸展、tau = [-0.7, -0.7] -> R = [0.7, 0.7]）
            R_init[m_s + 6, j_s + 1] = 0.7
            R_init[m_s + 6, j_s + 2] = 0.7
            # 筋7: 二関節筋・ハムストリングス様結合（大腿伸展かつ膝屈曲、tau = [0.7, 0.7] -> R = [-0.7, -0.7]）
            R_init[m_s + 7, j_s + 1] = -0.7
            R_init[m_s + 7, j_s + 2] = -0.7

        self.R = R_init.copy()

        # 2. 初期変位（プリテンション）ベクトル b (32次元)
        self.b = np.zeros(self.m, dtype=np.float64)

        # 3. 対数剛性 p_K および対数減衰 p_D (各32次元)
        # 正定値性を保証するため対数領域で最適化（K ~ 28.0 N/rad, D ~ 1.8 Ns/rad）
        self.p_K = np.full(self.m, np.log(28.0), dtype=np.float64)
        self.p_D = np.full(self.m, np.log(1.8), dtype=np.float64)

        # 4. 上位神経制御器の結合重み
        np.random.seed(seed)
        self.W_neural = np.random.randn(self.m, 4) * 0.05
        self.W_fb = np.random.randn(self.m, 6) * 0.02

        # トロット歩容（対角脚ペア同期）の初期結合設計:
        # 対角脚ペア1: FL (脚0), RR (脚3) - 基準位相 psi
        # 対角脚ペア2: FR (脚1), RL (脚2) - 逆位相 psi + pi
        w_sw = 2.4  # スイング期ゲイン
        w_st = 2.4  # スタンス期ゲイン

        for leg in [0, 3]:
            m_s = leg * 8
            # スイング相（sin成分）: 股関節屈筋・膝伸筋・大腿直筋を能動加振、拮抗筋を抑制
            self.W_neural[m_s + 2, 0] = w_sw
            self.W_neural[m_s + 5, 0] = w_sw * 0.7
            self.W_neural[m_s + 6, 0] = w_sw * 1.0
            self.W_neural[m_s + 3, 0] = -w_sw * 0.8
            self.W_neural[m_s + 4, 0] = -w_sw * 0.8

            # スタンス相（cos成分）: 股関節伸筋・膝屈筋・ハムストリングスを能動加振、拮抗筋を抑制
            self.W_neural[m_s + 3, 1] = w_st
            self.W_neural[m_s + 4, 1] = w_st * 0.7
            self.W_neural[m_s + 7, 1] = w_st * 1.0
            self.W_neural[m_s + 2, 1] = -w_st * 0.8
            self.W_neural[m_s + 5, 1] = -w_st * 0.8

        for leg in [1, 2]:
            m_s = leg * 8
            # スイング相（sin(psi + pi)成分）
            self.W_neural[m_s + 2, 2] = w_sw
            self.W_neural[m_s + 5, 2] = w_sw * 0.7
            self.W_neural[m_s + 6, 2] = w_sw * 1.0
            self.W_neural[m_s + 3, 2] = -w_sw * 0.8
            self.W_neural[m_s + 4, 2] = -w_sw * 0.8

            # スタンス相（cos(psi + pi)成分）
            self.W_neural[m_s + 3, 3] = w_st
            self.W_neural[m_s + 4, 3] = w_st * 0.7
            self.W_neural[m_s + 7, 3] = w_st * 1.0
            self.W_neural[m_s + 2, 3] = -w_st * 0.8
            self.W_neural[m_s + 5, 3] = -w_st * 0.8

        # 胴体姿勢フィードバック（左右ロール傾きに対する股関節外転・内転の自律復元）
        for leg in range(4):
            m_s = leg * 8
            is_left = (leg in [0, 2])
            sign = 1.0 if is_left else -1.0
            self.W_fb[m_s + 0, 0] = -0.3 * sign  # ロール角に応じた外転補正
            self.W_fb[m_s + 1, 0] = 0.3 * sign   # ロール角に応じた内転補正

        self.F_max = 24.0  # 最大能動収縮力 [N]

    def get_params(self) -> np.ndarray:
        """最適化対象の全800パラメータを平坦化ベクトルとして取得

        Returns:
            np.ndarray: 平坦化されたパラメータベクトル（次元: 800）。
        """
        return np.concatenate([
            (self.R * self.R_mask).flatten(),  # 32 x 12 = 384（ブロック対角マスク適用済）
            self.b,                            # 32
            self.p_K,                          # 32
            self.p_D,                          # 32
            self.W_neural.flatten(),           # 32 x 4 = 128
            self.W_fb.flatten()                # 32 x 6 = 192
        ])

    def set_params(self, params: np.ndarray) -> None:
        """平坦化パラメータベクトルから各内部行列・ベクトルを復元

        解剖学的ブロック対角マスク R_mask を乗算し、異なる脚間の交差結合を厳密にゼロに保つ。

        Args:
            params (np.ndarray): 800次元のパラメータベクトル。
        """
        idx = 0
        n_R = self.m * self.n
        self.R = params[idx : idx + n_R].reshape((self.m, self.n)) * self.R_mask
        idx += n_R

        self.b = params[idx : idx + self.m]
        idx += self.m

        self.p_K = params[idx : idx + self.m]
        idx += self.m

        self.p_D = params[idx : idx + self.m]
        idx += self.m

        n_Wn = self.m * 4
        self.W_neural = params[idx : idx + n_Wn].reshape((self.m, 4))
        idx += n_Wn

        n_Wfb = self.m * 6
        self.W_fb = params[idx : idx + n_Wfb].reshape((self.m, 6))

    def compute_torque(self, obs: dict) -> np.ndarray:
        """観測信号から関節トルク指令を計算

        計算手順:
            1. 上位神経回路: x_osc および s_imu から能動収縮力 f_active を算出。
            2. 幾何学的写像: 関節変位 q および速度 dq から仮想筋の伸長量と速度を算出。
            3. たるみ非線形性: 受動弾性項、減衰項、能動項を加算し ReLU で非負張力を決定。
            4. トルク変換: モーメントアーム行列の負転置 -R.T を乗算して関節トルクを出力。

        Args:
            obs (dict): 環境からの観測辞書。
                - "x_osc": 4次元の直交位相信号 [sin psi, cos psi, sin(psi+pi), cos(psi+pi)]。
                - "s_imu": 6次元の姿勢・固有受容信号 [roll, pitch, wx, wy, wz, z_kin - 0.30]。
                - "q_joints": 12次元の公称姿勢からの関節変位 [rad]。
                - "dq_joints": 12次元の関節角速度 [rad/s]。

        Returns:
            np.ndarray: 12次元の関節トルクベクトル tau [N*m]。
        """
        x_osc = obs["x_osc"]
        s_imu = obs["s_imu"]
        q = obs["q_joints"]
        dq = obs["dq_joints"]

        # 1. 上位極小神経回路の出力計算（能動筋収縮力）
        logits = self.W_neural @ x_osc + self.W_fb @ s_imu
        f_active = self.F_max * sigmoid(logits)

        # 2. 仮想剛性・減衰パラメータの復元（指数変換により正定値性を担保）
        K = np.exp(np.clip(self.p_K, 1.5, 5.5))
        D = np.exp(np.clip(self.p_D, -1.0, 2.5))

        # 3. 仮想筋伸長変位および伸長速度の幾何学的写像
        muscle_strain = self.R @ q + self.b
        muscle_strain_rate = self.R @ dq

        # 4. 厳密な非負筋張力の算出（ReLU によるたるみ・短縮アンローディング表現）
        total_muscle_tension = relu(K * muscle_strain + D * muscle_strain_rate + f_active)

        # 5. 仮想仕事の原理に基づく関節トルクへの変換
        tau = - self.R.T @ total_muscle_tension
        return tau


class MinimalPD:
    """アブレーション検証モデル: 極小神経制御器＋関節独立線形PD制御

    提案手法と同じ上位極小神経回路（位相振動子入力）を用いながら、
    下位アクチュエータ層として多関節腱結合を持たない関節独立型線形PD制御器を使用する。
    提案手法との公平な比較のため、同一の初期トロット対角歩容バイアスを持つ。

    数理定式化:
        delta_q_target = A_max * tanh(W_neural @ x_osc + W_fb @ s_imu + b)
        tau = Kp * (delta_q_target - q) - Kd * dq

    Attributes:
        n (int): 関節自由度数（12自由度）。
        W_neural (np.ndarray): 位相振動子信号に対する結合重み行列（形状: 12 x 4）。
        W_fb (np.ndarray): 胴体姿勢フィードバックに対する結合重み行列（形状: 12 x 6）。
        b (np.ndarray): 目標関節角度バイアス（形状: 12）。
        Kp (np.ndarray): 各関節の比例ゲイン（形状: 12）。
        Kd (np.ndarray): 各関節の微分ゲイン（形状: 12）。
        A_max (float): 最大関節目標変位振幅 [rad]（0.35 rad）。
    """

    def __init__(self, seed: int = 42):
        """Minimal-PD のパラメータ初期化

        Args:
            seed (int): 乱数シード。
        """
        self.n = 12
        np.random.seed(seed)
        self.W_neural = np.random.randn(self.n, 4) * 0.05
        self.W_fb = np.random.randn(self.n, 6) * 0.02
        self.b = np.zeros(self.n)
        self.Kp = np.full(self.n, 35.0)
        self.Kd = np.full(self.n, 1.8)
        self.A_max = 0.35  # 最大関節目標角度変位 [rad]

        # 提案手法と公平な初期対角トロット歩容バイアスの設定
        for leg in [0, 3]:
            j_s = leg * 3
            self.W_neural[j_s + 1, 0] = -1.2  # スイング期の大腿屈曲指令
            self.W_neural[j_s + 1, 1] = 1.2   # スタンス期の大腿伸展指令
            self.W_neural[j_s + 2, 0] = -1.0  # スイング期の膝伸展指令
            self.W_neural[j_s + 2, 1] = 1.0   # スタンス期の膝屈曲指令

        for leg in [1, 2]:
            j_s = leg * 3
            self.W_neural[j_s + 1, 2] = -1.2
            self.W_neural[j_s + 1, 3] = 1.2
            self.W_neural[j_s + 2, 2] = -1.0
            self.W_neural[j_s + 2, 3] = 1.0

    def get_params(self) -> np.ndarray:
        """平坦化された全156パラメータを取得

        Returns:
            np.ndarray: パラメータベクトル（次元: 156）。
        """
        return np.concatenate([
            self.W_neural.flatten(),  # 12 x 4 = 48
            self.W_fb.flatten(),      # 12 x 6 = 72
            self.b,                   # 12
            self.Kp,                  # 12
            self.Kd                   # 12
        ])

    def set_params(self, params: np.ndarray) -> None:
        """平坦化パラメータベクトルから各内部変数を復元

        Args:
            params (np.ndarray): 156次元のパラメータベクトル。
        """
        idx = 0
        n_wn = self.n * 4
        self.W_neural = params[idx : idx + n_wn].reshape((self.n, 4))
        idx += n_wn

        n_wfb = self.n * 6
        self.W_fb = params[idx : idx + n_wfb].reshape((self.n, 6))
        idx += n_wfb

        self.b = params[idx : idx + self.n]
        idx += self.n

        self.Kp = np.clip(params[idx : idx + self.n], 10.0, 100.0)
        idx += self.n

        self.Kd = np.clip(params[idx : idx + self.n], 0.5, 10.0)

    def compute_torque(self, obs: dict) -> np.ndarray:
        """独立PD制御則による関節トルクの算出

        Args:
            obs (dict): 環境からの観測辞書。

        Returns:
            np.ndarray: 12次元の関節トルクベクトル [N*m]。
        """
        x_osc = obs["x_osc"]
        s_imu = obs["s_imu"]
        q = obs["q_joints"]
        dq = obs["dq_joints"]

        delta_q_target = self.A_max * np.tanh(self.W_neural @ x_osc + self.W_fb @ s_imu + self.b)
        tau = self.Kp * (delta_q_target - q) - self.Kd * dq
        return tau


class BaselineMLP:
    """従来型深層強化学習ベースライン: 3層MLP＋独立PD制御

    環境状態全般（観測34次元）を入力とし、3層パーセプトロンによって各関節の目標角度を出力する従来構成。
    下位アクチュエータ層は独立関節PD制御器を用いる。

    ネットワーク構成:
        入力層 (34) -> 隠れ層1 (64, tanh) -> 隠れ層2 (64, tanh) -> 出力層 (12, tanh) -> PD制御
        総パラメータ数: 34*64 + 64 + 64*64 + 64 + 64*12 + 12 = 7,180 パラメータ。

    Attributes:
        dim_in (int): 入力次元数（34次元）。
        dim_h (int): 隠れ層ノード数（64ノード）。
        dim_out (int): 出力次元数（12次元）。
        W1, W2, W3 (np.ndarray): 各層の重み行列。
        b1, b2, b3 (np.ndarray): 各層のバイアスベクトル。
        Kp, Kd (float): PD制御の比例・微分ゲイン。
    """

    def __init__(self, seed: int = 42):
        """Baseline-MLP の重み・バイアス初期化

        Args:
            seed (int): 乱数シード。
        """
        self.dim_in = 34
        self.dim_h = 64
        self.dim_out = 12

        np.random.seed(seed)
        self.W1 = np.random.randn(self.dim_h, self.dim_in) * np.sqrt(2.0 / self.dim_in)
        self.b1 = np.zeros(self.dim_h)
        self.W2 = np.random.randn(self.dim_h, self.dim_h) * np.sqrt(2.0 / self.dim_h)
        self.b2 = np.zeros(self.dim_h)
        self.W3 = np.random.randn(self.dim_out, self.dim_h) * np.sqrt(2.0 / self.dim_h) * 0.1
        self.b3 = np.zeros(self.dim_out)

        self.Kp = 35.0
        self.Kd = 1.8
        self.A_max = 0.35

    def get_params(self) -> np.ndarray:
        """全7,180パラメータを平坦化ベクトルとして取得

        Returns:
            np.ndarray: パラメータベクトル（次元: 7,180）。
        """
        return np.concatenate([
            self.W1.flatten(), self.b1,
            self.W2.flatten(), self.b2,
            self.W3.flatten(), self.b3
        ])

    def set_params(self, params: np.ndarray) -> None:
        """平坦化パラメータベクトルからMLP重み・バイアスを復元

        Args:
            params (np.ndarray): 7,180次元のパラメータベクトル。
        """
        idx = 0
        s1 = self.dim_h * self.dim_in
        self.W1 = params[idx : idx + s1].reshape((self.dim_h, self.dim_in))
        idx += s1
        self.b1 = params[idx : idx + self.dim_h]
        idx += self.dim_h

        s2 = self.dim_h * self.dim_h
        self.W2 = params[idx : idx + s2].reshape((self.dim_h, self.dim_h))
        idx += s2
        self.b2 = params[idx : idx + self.dim_h]
        idx += self.dim_h

        s3 = self.dim_out * self.dim_h
        self.W3 = params[idx : idx + s3].reshape((self.dim_out, self.dim_h))
        idx += s3
        self.b3 = params[idx : idx + self.dim_out]

    def compute_torque(self, obs: dict) -> np.ndarray:
        """MLPによる順伝播計算および独立PD制御によるトルク算出

        Args:
            obs (dict): 環境からの観測辞書。

        Returns:
            np.ndarray: 12次元の関節トルクベクトル [N*m]。
        """
        x = np.concatenate([
            obs["x_osc"],
            obs["s_imu"],
            obs["q_joints"],
            obs["dq_joints"]
        ])
        h1 = np.tanh(self.W1 @ x + self.b1)
        h2 = np.tanh(self.W2 @ h1 + self.b2)
        delta_q_target = self.A_max * np.tanh(self.W3 @ h2 + self.b3)

        tau = self.Kp * (delta_q_target - obs["q_joints"]) - self.Kd * obs["dq_joints"]
        return tau
