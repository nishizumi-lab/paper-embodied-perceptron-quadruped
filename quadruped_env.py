"""MuJoCo四脚ロボット（Unitree Go2相当）シミュレーション環境ラッパーモジュール

本モジュールは、MuJoCo物理シミュレータ上で12自由度電動四脚ロボットの力学挙動を計算し、
上位制御器に対する観測信号の抽出、QDDアクチュエータのトルク・回転数飽和特性を考慮した
トルク指令の適用、接地反力（GRF）に基づく位相エントレインメント（Tegotaeフィードバック）、
外乱インパルスの印加、および胴体座標系（Body frame）における強化学習報酬関数の算出を行う
環境クラス QuadrupedEnv を提供する。
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

    関節可動範囲およびアクチュエータ仕様:
        公称関節角度: 各脚 [Hip AA = 0.0, Hip FE = 0.8, Knee FE = -1.6] rad。
        ピークトルク限界: Hip AA [-24, 24] N*m, Hip FE [-24, 24] N*m, Knee FE [-45, 45] N*m。
        無負荷最大角速度（逆起電力限界）: Hip AA 30.0 rad/s, Hip FE 30.0 rad/s, Knee FE 20.0 rad/s。

    状態推定および感覚フィードバック:
        1. 固有受容感覚（関節エンコーダ＋IMU姿勢角）による順運動学ベースの胴体高さ推定 z_kin。
        2. ワールド座標系並進速度からロボット胴体座標系（Body frame）並進速度へのクォータニオン回転変換。
        3. 足先接地反力（GRF）を用いた歩行位相振動子の位相リセット・同調機構（Tegotae効果）。

    Attributes:
        model (mujoco.MjModel): MuJoCoモデル構造体。
        data (mujoco.MjData): MuJoCo力学状態データ構造体。
        dt_sim (float): シミュレーション積分刻み幅 [s]。
        decimation (int): 制御間引きステップ数。
        dt_ctrl (float): 上位制御周期 [s]。
        max_steps (int): 1エピソードの最大ステップ数。
        step_count (int): 現在のエピソード経過ステップ数。
        q_nominal (np.ndarray): 公称直立関節角度ベクトル（12次元）。
        tau_peak (np.ndarray): 各関節のピークトルク上限ベクトル（12次元）[N*m]。
        omega_max (np.ndarray): 各関節の無負荷最大角速度ベクトル（12次元）[rad/s]。
        foot_geom_ids (list[int]): 4脚足先球体ジオメトリのMuJoCo IDリスト。
        trot_freq (float): トロット歩容の基本周波数 [Hz]（1.6 Hz）。
        sigma_tegotae (float): 接地反力による位相調整ゲイン [rad/s]（0.35）。
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

        # QDDアクチュエータのピークトルク上限 [N*m] および最大角速度 [rad/s]
        self.tau_peak = np.array([24.0, 24.0, 45.0] * 4, dtype=np.float64)
        self.omega_max = np.array([30.0, 30.0, 20.0] * 4, dtype=np.float64)

        # リンク長パラメータ（順運動学による固有受容胴体高さ推定用）
        self.l_thigh = 0.213  # 大腿リンク長 [m]
        self.l_calf = 0.213   # 下腿リンク長 [m]
        self.r_foot = 0.025   # 足先半径 [m]

        # 4脚の足先ジオメトリIDの取得（接地反力抽出用）
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self.foot_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in foot_names
        ]

        # トロット歩行目標仕様および接地反力位相調整（Tegotae）ゲイン
        self.trot_freq = 1.6       # 基本歩容周波数 1.6 Hz
        self.sigma_tegotae = 0.35  # 接地反力による位相エントレインメント強度
        self.v_target = 0.6        # 目標前進速度 0.6 m/s
        self.phase = 0.0           # 歩行位相初期値

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

    def _estimate_kinematic_height(self, roll: float, pitch: float) -> float:
        """固有受容感覚（関節エンコーダ角度とIMU傾き角）に基づく胴体高さの順運動学推定

        実機IMUではワールド座標系の絶対高度 z を直接計測できないため、
        4脚の関節角度 (q_aa, q_fe, q_kn) から各足先の胴体に対する垂直距離を計算し、
        支持脚（最も伸展している上位2脚の平均）およびロール・ピッチ傾斜角から
        地面に対する胴体高さ z_kin をオンボード推定する。

        数理計算式:
            z_leg,l = (L_1 cos(q_fe,l) + L_2 cos(q_fe,l + q_kn,l)) cos(q_aa,l) + r_foot
            z_kin = cos(roll) * cos(pitch) * mean(top2(z_leg))

        Args:
            roll (float): 胴体ロール角 [rad]。
            pitch (float): 胴体ピッチ角 [rad]。

        Returns:
            float: 順運動学により推定された胴体高さ z_kin [m]。
        """
        q_abs = self.data.qpos[7:19]
        leg_heights = np.zeros(4, dtype=np.float64)
        for leg in range(4):
            q_aa = q_abs[leg * 3 + 0]
            q_fe = q_abs[leg * 3 + 1]
            q_kn = q_abs[leg * 3 + 2]
            z_ext = (self.l_thigh * np.cos(q_fe) + self.l_calf * np.cos(q_fe + q_kn)) * np.cos(q_aa) + self.r_foot
            leg_heights[leg] = z_ext

        # 接地支持脚に相当する上位2脚の平均伸長距離に姿勢傾斜投影を乗算
        sorted_h = np.sort(leg_heights)
        stance_h = 0.5 * (sorted_h[2] + sorted_h[3])
        z_kin = float(np.cos(roll) * np.cos(pitch) * stance_h)
        return z_kin

    def _get_foot_normal_forces(self) -> np.ndarray:
        """4脚それぞれの足先が地面から受ける垂直接地反力（GRF）を抽出

        Returns:
            np.ndarray: 4次元ベクトル [N_FL, N_FR, N_RL, N_RR]（単位: [N]、非負）。
        """
        foot_forces = np.zeros(4, dtype=np.float64)
        c_force = np.zeros(6, dtype=np.float64)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            for leg_idx, fg_id in enumerate(self.foot_geom_ids):
                if g1 == fg_id or g2 == fg_id:
                    mujoco.mj_contactForce(self.model, self.data, i, c_force)
                    foot_forces[leg_idx] += abs(c_force[0])
        return foot_forces

    def _get_body_frame_velocity(self) -> tuple[float, float, float]:
        """ワールド座標系の胴体並進速度をロボット胴体座標系（Body frame）へ座標変換

        MuJoCoの freejoint における qvel[0:3] はワールド座標系で表現されるため、
        ロボットがヨー回転した際にも正確な前進速度 vx_body と横滑り速度 vy_body を得るには、
        胴体姿勢クォータニオンから算出した回転行列 R_wb の転置を乗算する必要がある。

        数理計算式:
            v_body = R_wb^T @ v_world

        Returns:
            tuple[float, float, float]: 胴体座標系における (vx_body, vy_body, vz_body) [m/s]。
        """
        quat = self.data.qpos[3:7]
        rot_mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot_mat, quat)
        R_wb = rot_mat.reshape((3, 3))
        v_world = self.data.qvel[0:3]
        v_body = R_wb.T @ v_world
        return float(v_body[0]), float(v_body[1]), float(v_body[2])

    def _get_obs(self) -> dict:
        """現在のロボット力学状態から実機可観測なセンサ信号ベクトルを抽出

        クォータニオンからロール・ピッチ角への変換式:
            roll = arctan2(2(w*x + y*z), 1 - 2(x^2 + y^2))
            pitch = arcsin(clip(2(w*y - z*x), -1, 1))

        直交位相基底ベクトル:
            x_osc = [sin(psi), cos(psi), sin(psi + pi), cos(psi + pi)]^T

        Returns:
            dict: 制御器入力用の観測信号辞書。
                - "s_imu": 6次元姿勢・固有受容ベクトル [roll, pitch, wx, wy, wz, z_kin - 0.30]。
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

        # ローカル胴体座標系の3軸角速度および順運動学に基づく推定胴体高さ偏差
        omega = self.data.qvel[3:6]
        z_kin = self._estimate_kinematic_height(roll, pitch)
        s_imu = np.array([roll, pitch, omega[0], omega[1], omega[2], z_kin - 0.30], dtype=np.float64)

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
            force_y (float): ロボット胴体横方向に加える外力 [N]。
            duration_sec (float): 外力印加時間 [s]（デフォルト: 0.1秒）。
        """
        self.perturb_force = np.array([0.0, force_y, 0.0])
        self.perturb_steps_left = int(duration_sec / self.dt_sim)

    def step(self, tau: np.ndarray) -> tuple[dict, float, bool, dict]:
        """シミュレーションを1制御ステップ（0.008秒）進める

        処理フロー:
            1. ブラシレスQDDモータの逆起電力（Back-EMF）特性に基づくトルク・速度飽和制約を適用。
            2. 間引きループ（decimation = 4）内で物理積分 mj_step を実行し、外力を適用。
            3. 足先接地反力（GRF）に基づくTegotae位相フィードバックを伴う基本位相 psi の更新。
            4. 胴体座標系（Body frame）における前進速度・横滑り速度と姿勢に基づく報酬計算。
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
        # 制御周期内の物理微小積分（4ステップ = 0.008秒）
        for _ in range(self.decimation):
            if self.perturb_steps_left > 0:
                self.data.xfrc_applied[1, :3] = self.perturb_force
                self.perturb_steps_left -= 1
            else:
                self.data.xfrc_applied[1, :3] = 0.0

            # 関節角速度に応じたQDDモータの逆起電力トルク・回転数飽和曲線:
            # tau_max(dq) = tau_peak * clip(1 - |dq| / omega_max, 0.15, 1.0)
            dq_curr = self.data.qvel[6:18]
            speed_factor = np.clip(1.0 - np.abs(dq_curr) / self.omega_max, 0.15, 1.0)
            dyn_tau_limit = self.tau_peak * speed_factor
            tau_applied = np.clip(tau, -dyn_tau_limit, dyn_tau_limit)

            self.data.ctrl[:] = tau_applied
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        # 足先接地反力（GRF）に基づく歩行位相の同調更新（Tegotae位相リセット効果）:
        # 対角ペア1 (FL=0, RR=3) と 対角ペア2 (FR=1, RL=2) の正規化荷重差を算出
        grf = self._get_foot_normal_forces()
        pair1_load = np.clip((grf[0] + grf[3]) / 120.0, 0.0, 1.5)
        pair2_load = np.clip((grf[1] + grf[2]) / 120.0, 0.0, 1.5)
        phase_mod = - self.sigma_tegotae * (pair1_load - pair2_load) * np.sin(self.phase)
        omega_eff = 2.0 * np.pi * self.trot_freq + phase_mod
        self.phase = (self.phase + omega_eff * self.dt_ctrl) % (2.0 * np.pi)

        obs = self._get_obs()

        # ロボット胴体座標系（Body frame）における並進速度および姿勢変量
        vx, vy, _ = self._get_body_frame_velocity()
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

        # 3. 胴体座標系における側方ドリフト速度ペナルティ
        r_lat = - 2.0 * (vy ** 2)

        # 4. モータートルク二乗和（消費エネルギー）ペナルティ
        r_tau = - 0.0003 * np.sum(tau_applied ** 2)

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
