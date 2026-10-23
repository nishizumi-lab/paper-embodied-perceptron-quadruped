"""MuJoCo四脚ロボット（Unitree Go2相当）シミュレーション環境ラッパーモジュール

本モジュールは、MuJoCo物理シミュレータ上で12自由度電動四脚ロボットの力学挙動を計算し、
上位制御器に対する観測信号の抽出、トルク指令の適用、外乱インパルスの印加、
および強化学習用の報酬関数の算出を行う環境クラス QuadrupedEnv を提供する。
"""

import os
import mujoco
import numpy as np


class QuadrupedEnv:
    """MuJoCo四脚ロボット歩行シミュレーション環境

    1脚あたり3自由度（股関節外転・内転、股関節屈曲・伸展、膝関節屈曲・伸展）、計12自由度を持つ
    四脚歩行ロボット（Unitree Go2モデル）の力学環境をカプセル化する。

    時間管理パラメータ:
        dt_sim: 物理シミュレーション積分刻み幅（0.002秒 = 500 Hz）。
        decimation: 制御間引き数（4ステップ）。
        dt_ctrl: 制御器更新周期（0.002 * 4 = 0.008秒 = 125 Hz）。

    関節可動範囲および公称直立姿勢:
        公称関節角度: 各脚 [Hip AA = 0.0, Hip FE = 0.8, Knee FE = -1.6] rad。
        トルク限界: Hip AA [-24, 24] N*m, Hip FE [-24, 24] N*m, Knee FE [-45, 45] N*m。

    Attributes:
        model (mujoco.MjModel): MuJoCoモデル構造体。
        data (mujoco.MjData): MuJoCo力学状態データ構造体。
        dt_sim (float): シミュレーション積分刻み幅 [s]。
        decimation (int): 制御間引きステップ数。
        dt_ctrl (float): 上位制御周期 [s]。
        max_steps (int): 1エピソードの最大ステップ数。
        step_count (int): 現在のエピソード経過ステップ数。
        q_nominal (np.ndarray): 公称直立関節角度ベクトル（12次元）。
        trot_freq (float): トロット歩容の基本周波数 [Hz]（1.6 Hz）。
        v_target (float): 目標前進速度 [m/s]（0.6 m/s）。
        phase (float): 現在の歩行位相 psi [rad]（[0, 2*pi)）。
    """

    def __init__(self, xml_path: str = None, dt: float = 0.002, decimation: int = 4, max_steps: int = 400):
        """シミュレーション環境の初期化

        Args:
            xml_path (str, optional): MuJoCo MJCFモデルXMLファイルのパス。Noneの場合は同ディレクトリの go2_model.xml を参照。
            dt (float): 物理シミュレーション積分周期 [s]（デフォルト: 0.002秒）。
            decimation (int): 制御周期に対する間引き比率（デフォルト: 4）。
            max_steps (int): 1エピソードの最大制御ステップ数（デフォルト: 400ステップ）。
        """
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "go2_model.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.dt_sim = self.model.opt.timestep
        self.decimation = decimation
        self.dt_ctrl = self.dt_sim * self.decimation  # 0.008s -> 125 Hz
        self.max_steps = max_steps
        self.step_count = 0

        # 直立基準姿勢における各脚の公称関節角度 [rad]:
        # Hip AA (外転・内転) = 0.0 rad
        # Hip FE (屈曲・伸展) = 0.8 rad（大腿やや後傾）
        # Knee FE (膝屈曲・伸展) = -1.6 rad（膝屈曲位）
        self.q_nominal = np.array([
            0.0, 0.8, -1.6,   # 前左脚 (FL)
            0.0, 0.8, -1.6,   # 前右脚 (FR)
            0.0, 0.8, -1.6,   # 後左脚 (RL)
            0.0, 0.8, -1.6    # 後右脚 (RR)
        ], dtype=np.float64)

        # トロット歩行目標仕様
        self.trot_freq = 1.6  # 基本歩容周波数 1.6 Hz
        self.v_target = 0.6   # 目標前進速度 0.6 m/s
        self.phase = 0.0      # 歩行位相初期値

        # 外乱インパルス用変数
        self.perturb_force = np.zeros(3)
        self.perturb_steps_left = 0

    def reset(self, randomize: bool = True) -> dict:
        """環境状態のリセット

        ロボットの胴体位置、姿勢クォータニオン、および関節角度を直立公称姿勢へ初期化する。

        Args:
            randomize (bool): 初期関節角度に微小ノイズ（[-0.03, 0.03] rad）を付加するか否か。

        Returns:
            dict: 初期観測辞書（_get_obs() の戻り値）。
        """
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.phase = 0.0
        self.perturb_force[:] = 0.0
        self.perturb_steps_left = 0

        # 胴体の基準空間位置・姿勢の設定（地上高 0.315 m, 水平姿勢）
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = 0.315
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 単位クォータニオン [w, x, y, z]

        # 関節角度の初期化
        noise = np.random.uniform(-0.03, 0.03, size=12) if randomize else 0.0
        self.data.qpos[7:19] = self.q_nominal + noise
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _get_obs(self) -> dict:
        """現在のロボット力学状態から観測信号ベクトルを抽出

        クォータニオンからロール・ピッチ角への変換式:
            roll = arctan2(2(w*x + y*z), 1 - 2(x^2 + y^2))
            pitch = arcsin(clip(2(w*y - z*x), -1, 1))

        直交位相基底ベクトル:
            x_osc = [sin(psi), cos(psi), sin(psi + pi), cos(psi + pi)]^T

        Returns:
            dict: 制御器入力用の観測信号辞書。
                - "s_imu": 6次元姿勢ベクトル [roll, pitch, wx, wy, wz, delta_z]。
                - "x_osc": 4次元トロット直交位相信号。
                - "q_joints": 12次元の公称姿勢からの関節変位 [rad]。
                - "dq_joints": 12次元の関節角速度 [rad/s]。
                - "phase": 現在の基本位相 psi [rad]。
        """
        # クォータニオン [w, x, y, z] からオイラー角（ロール・ピッチ）を算出
        q = self.data.qpos[3:7]
        w, x, y, z = q[0], q[1], q[2], q[3]
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))

        # 胴体角速度および基準高さからの偏差
        omega = self.data.qvel[3:6]
        z_pos = self.data.qpos[2]
        s_imu = np.array([roll, pitch, omega[0], omega[1], omega[2], z_pos - 0.30], dtype=np.float64)

        # トロット歩容用の直交周期基底（FL/RRの対角ペアとFR/RLの対角ペアに対応）
        psi = self.phase
        x_osc = np.array([
            np.sin(psi),
            np.cos(psi),
            np.sin(psi + np.pi),
            np.cos(psi + np.pi)
        ], dtype=np.float64)

        # 公称姿勢からの関節変位および角速度
        q_joints = self.data.qpos[7:19] - self.q_nominal
        dq_joints = self.data.qvel[6:18]

        return {
            "s_imu": s_imu,
            "x_osc": x_osc,
            "q_joints": q_joints,
            "dq_joints": dq_joints,
            "phase": self.phase
        }

    def apply_lateral_impulse(self, force_y: float, duration_sec: float = 0.1) -> None:
        """外乱耐性試験用の横方向インパルス外力の設定

        Args:
            force_y (float): ロボット胴体横方向（Y軸方向）に加える外力 [N]。
            duration_sec (float): 外力印加時間 [s]（デフォルト: 0.1秒）。
        """
        self.perturb_force = np.array([0.0, force_y, 0.0])
        self.perturb_steps_left = int(duration_sec / self.dt_sim)

    def step(self, tau: np.ndarray) -> tuple[dict, float, bool, dict]:
        """シミュレーションを1制御ステップ（0.008秒）進める

        処理フロー:
            1. 各関節のトルク限界に基づきトルク指令 tau をクリップ。
            2. 間引きループ（decimation = 4）内で物理積分 mj_step を実行し、外力を適用。
            3. 基本位相 psi を時間積分で更新。
            4. 前進速度、姿勢傾き、エネルギー消費に基づく報酬関数を計算。
            5. 転倒判定（高さ・傾き）を行い終了フラグを判定。

        Args:
            tau (np.ndarray): 12次元の目標関節トルク指令ベクトル [N*m]。

        Returns:
            tuple[dict, float, bool, dict]:
                - obs (dict): 更新後の観測信号辞書。
                - reward (float): 当該ステップの即時報酬。
                - done (bool): エピソード終了フラグ。
                - info (dict): 評価・デバッグ用情報辞書。
        """
        # MuJoCo XMLモデル定義に整合する各関節のトルク制限:
        # Hip AA: [-24, 24] N*m, Hip FE: [-24, 24] N*m, Knee FE: [-45, 45] N*m
        tau_limit = np.array([24.0, 24.0, 45.0] * 4, dtype=np.float64)
        tau = np.clip(tau, -tau_limit, tau_limit)

        # 制御周期内の物理微小積分（4ステップ = 0.008秒）
        for _ in range(self.decimation):
            if self.perturb_steps_left > 0:
                self.data.xfrc_applied[1, :3] = self.perturb_force
                self.perturb_steps_left -= 1
            else:
                self.data.xfrc_applied[1, :3] = 0.0

            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        # 位相の時間発展式: psi = (psi + 2*pi * f_trot * dt_ctrl) mod 2*pi
        self.phase = (self.phase + 2.0 * np.pi * self.trot_freq * self.dt_ctrl) % (2.0 * np.pi)

        obs = self._get_obs()

        # 胴体の線速度および姿勢変量
        vx = self.data.qvel[0]
        vy = self.data.qvel[1]
        roll = obs["s_imu"][0]
        pitch = obs["s_imu"][1]
        height = self.data.qpos[2]

        # 報酬関数の各コンポーネント算出:
        # 1. 前進速度追従項（目標速度 0.6 m/s でピークとなるガウス型報酬、後退時はペナルティ）
        r_vel = 2.0 * np.exp(- 4.0 * (vx - self.v_target) ** 2)
        if vx < 0:
            r_vel -= 1.0 * abs(vx)

        # 2. 姿勢水平性および目標高さ安定項
        r_ori = - 2.0 * (roll ** 2 + pitch ** 2)
        r_height = - 8.0 * ((height - 0.30) ** 2)

        # 3. 側方ドリフト速度ペナルティ
        r_lat = - 2.0 * (vy ** 2)

        # 4. モータートルク二乗和（消費エネルギー）ペナルティ
        r_tau = - 0.0003 * np.sum(tau ** 2)

        # 5. 生存継続ボーナス
        r_alive = 1.0

        reward = r_vel + r_ori + r_height + r_lat + r_tau + r_alive

        # 転倒・異常姿勢によるエピソード終了判定
        done = False
        if height < 0.17 or height > 0.45 or abs(roll) > 0.65 or abs(pitch) > 0.65:
            done = True
            reward -= 5.0  # 転倒ペナルティ

        if self.step_count >= self.max_steps:
            done = True

        info = {
            "vx": vx,
            "vy": vy,
            "height": height,
            "roll": roll,
            "pitch": pitch,
            "step_count": self.step_count
        }
        return obs, reward, done, info
