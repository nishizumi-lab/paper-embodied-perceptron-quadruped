"""身体化パーセプトロン表現に基づく四脚歩行制御モデルモジュール

本モジュールは、腱駆動筋骨格系の静力学モデルと3層パーセプトロンの数理的等価性に基づき、
電動四脚ロボットのアクチュエータ空間に仮想多関節腱ネットワークを構築する
「仮想筋骨格レイヤー（ProposedVM）」、および比較検証用の「関節独立PD制御（MinimalPD）」、
「多層パーセプトロンベースライン（BaselineMLP）」を定義する。
"""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """シグモイド活性化関数

    ロジット入力を [0, 1] の確率的・有界な値域へ変換する。
    数値的オーバーフローを防止するため、入力を [-15.0, 15.0] にクリップする。

    Args:
        x (np.ndarray): 入力配列。

    Returns:
        np.ndarray: シグモイド関数の出力配列。
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


def relu(x: np.ndarray) -> np.ndarray:
    """正規化線形関数（ReLU）

    腱・ワイヤーの片側拘束（引張力のみを伝達し、圧縮時にはたるむSlack特性）を表現する。

    Args:
        x (np.ndarray): 入力プレ張力配列。

    Returns:
        np.ndarray: 非負にクリップされた張力配列（max(0, x)）。
    """
    return np.maximum(0.0, x)


class ProposedVM:
    """身体化パーセプトロン仮想筋骨格レイヤーおよび極小神経制御器

    仮想仕事の原理および身体化パーセプトロン表現に基づく仮想インピーダンス制御モデル。
    各仮想筋の発生張力を常に非負（引張力のみ）とし、関節トルクをモーメントアーム行列の負転置で写像する。

    数理定式化:
        1. 幾何学的筋伸長変位: Delta_l = R @ q + b
        2. 幾何学的筋伸長速度: Delta_dot_l = R @ dq
        3. 上位神経能動収縮力: f_active = F_max * sigmoid(W_neural @ x_osc + W_fb @ s_imu)
        4. 非負仮想筋張力: f = relu(K * Delta_l + D * Delta_dot_l + f_active)
        5. 関節出力トルク: tau = - R.T @ f

    Attributes:
        m (int): 仮想筋の総本数（32本、1脚あたり8本）。
        n (int): 関節自由度の総数（12自由度、1脚あたり3関節）。
        R (np.ndarray): 仮想モーメントアーム行列（形状: 32 x 12）。
        b (np.ndarray): 初期変位（プリテンション）ベクトル（形状: 32）。
        p_K (np.ndarray): 対数剛性パラメータベクトル（形状: 32、K = exp(p_K)）。
        p_D (np.ndarray): 対数減衰パラメータベクトル（形状: 32、D = exp(p_D)）。
        W_neural (np.ndarray): 位相振動子信号に対する結合重み行列（形状: 32 x 4）。
        W_fb (np.ndarray): 胴体姿勢フィードバックに対する結合重み行列（形状: 32 x 6）。
        F_max (float): 各仮想筋の最大能動収縮力 [N]（本実装では 24.0 N）。
    """

    def __init__(self, seed: int = 42):
        """モデルパラメータの初期化

        各脚8本の仮想筋（単関節筋6本、二関節筋2本）の解剖学的モーメントアーム符号、
        相反神経支配（Reciprocal Inhibition）を考慮した初期トロット歩容結合、
        および対数剛性・減衰パラメータを設定する。

        Args:
            seed (int): 乱数シード。
        """
        self.m = 32  # 仮想筋数: 1脚あたり8本 x 4脚 = 32本
        self.n = 12  # 関節数: 1脚あたり3関節 x 4脚 = 12関節

        # 1. 仮想モーメントアーム行列 R の初期化（32 x 12）
        # トルク変換式 tau = - R.T @ f （f >= 0）に基づく符号規約:
        # 正トルク（tau > 0）を発生させたい場合: -R.T > 0 より R = -1.0
        # 負トルク（tau < 0）を発生させたい場合: -R.T < 0 より R = +1.0
        R_init = np.zeros((self.m, self.n))
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
        self.b = np.zeros(self.m)

        # 3. 対数剛性 p_K および対数減衰 p_D (各32次元)
        # 正定値性を保証するため対数領域で最適化（K ~ 28.0 N/rad, D ~ 1.8 Ns/rad）
        self.p_K = np.full(self.m, np.log(28.0))
        self.p_D = np.full(self.m, np.log(1.8))

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
            self.R.flatten(),           # 32 x 12 = 384
            self.b,                     # 32
            self.p_K,                   # 32
            self.p_D,                   # 32
            self.W_neural.flatten(),    # 32 x 4 = 128
            self.W_fb.flatten()         # 32 x 6 = 192
        ])

    def set_params(self, params: np.ndarray) -> None:
        """平坦化パラメータベクトルから各内部行列・ベクトルを復元

        Args:
            params (np.ndarray): 800次元のパラメータベクトル。
        """
        idx = 0
        n_R = self.m * self.n
        self.R = params[idx : idx + n_R].reshape((self.m, self.n))
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
                - "s_imu": 6次元の姿勢信号 [roll, pitch, wx, wy, wz, delta_z]。
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

        # 4. 厳密な非負筋張力の算出（ReLU によるたるみ表現）
        total_muscle_tension = relu(K * muscle_strain + D * muscle_strain_rate + f_active)

        # 5. 仮想仕事の原理に基づく関節トルクへの変換
        tau = - self.R.T @ total_muscle_tension
        return tau


class MinimalPD:
    """アブレーション検証モデル: 極小神経制御器＋関節独立線形PD制御

    提案手法と同じ上位極小神経回路（位相振動子入力）を用いながら、
    下位アクチュエータ層として多関節腱結合を持たない関節独立型線形PD制御器（R = I 固定）を使用する。
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
