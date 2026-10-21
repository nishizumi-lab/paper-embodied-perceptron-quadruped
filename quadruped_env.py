import os
import mujoco
import numpy as np

class QuadrupedEnv:
    def __init__(self, xml_path=None, dt=0.002, decimation=4, max_steps=400):
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "go2_model.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.dt_sim = self.model.opt.timestep
        self.decimation = decimation
        self.dt_ctrl = self.dt_sim * self.decimation  # 0.008s -> 125Hz
        self.max_steps = max_steps
        self.step_count = 0
        
        # Nominal joint angles for standing: Hip AA = 0.0, Hip FE = 0.8, Knee FE = -1.6
        self.q_nominal = np.array([
            0.0, 0.8, -1.6,   # FL
            0.0, 0.8, -1.6,   # FR
            0.0, 0.8, -1.6,   # RL
            0.0, 0.8, -1.6    # RR
        ], dtype=np.float64)
        
        # Target gait parameters
        self.trot_freq = 1.6  # Trot frequency 1.6 Hz
        self.v_target = 0.6   # Target forward velocity 0.6 m/s
        self.phase = 0.0
        
        # Disturbance
        self.perturb_force = np.zeros(3)
        self.perturb_steps_left = 0

    def reset(self, randomize=True):
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.phase = 0.0
        self.perturb_force[:] = 0
        self.perturb_steps_left = 0
        
        # Set base pose
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = 0.315
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # quat
        
        # Set joint angles
        noise = np.random.uniform(-0.03, 0.03, size=12) if randomize else 0.0
        self.data.qpos[7:19] = self.q_nominal + noise
        self.data.qvel[:] = 0.0
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):
        # IMU extraction: orientation (roll, pitch from quaternion)
        q = self.data.qpos[3:7]  # w, x, y, z
        w, x, y, z = q[0], q[1], q[2], q[3]
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        
        omega = self.data.qvel[3:6]  # angular velocity wx, wy, wz
        z_pos = self.data.qpos[2]
        
        s_imu = np.array([roll, pitch, omega[0], omega[1], omega[2], z_pos - 0.30], dtype=np.float64)
        
        # Oscillator signals for trot gait: FL & RR vs FR & RL
        psi = self.phase
        x_osc = np.array([
            np.sin(psi),
            np.cos(psi),
            np.sin(psi + np.pi),
            np.cos(psi + np.pi)
        ], dtype=np.float64)
        
        # Joint states
        q_joints = self.data.qpos[7:19] - self.q_nominal
        dq_joints = self.data.qvel[6:18]
        
        return {
            "s_imu": s_imu,
            "x_osc": x_osc,
            "q_joints": q_joints,
            "dq_joints": dq_joints,
            "phase": self.phase
        }

    def apply_lateral_impulse(self, force_y, duration_sec=0.1):
        self.perturb_force = np.array([0.0, force_y, 0.0])
        self.perturb_steps_left = int(duration_sec / self.dt_sim)

    def step(self, tau):
        # Clip torque limits (-45 to 45 N*m)
        tau = np.clip(tau, -45.0, 45.0)
        
        for _ in range(self.decimation):
            if self.perturb_steps_left > 0:
                self.data.xfrc_applied[1, :3] = self.perturb_force
                self.perturb_steps_left -= 1
            else:
                self.data.xfrc_applied[1, :3] = 0.0
                
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)
            
        self.step_count += 1
        self.phase = (self.phase + 2.0 * np.pi * self.trot_freq * self.dt_ctrl) % (2.0 * np.pi)
        
        obs = self._get_obs()
        
        # Base velocities
        vx = self.data.qvel[0]
        vy = self.data.qvel[1]
        roll = obs["s_imu"][0]
        pitch = obs["s_imu"][1]
        height = self.data.qpos[2]
        
        # Reward components
        # 1. Forward velocity reward (peaked at v_target = 0.6 m/s)
        r_vel = 2.0 * np.exp(- 4.0 * (vx - self.v_target) ** 2)
        if vx < 0:
            r_vel -= 1.0 * abs(vx)  # Penalty for moving backward
            
        # 2. Orientation & Height stability
        r_ori = - 2.0 * (roll ** 2 + pitch ** 2)
        r_height = - 8.0 * ((height - 0.30) ** 2)
        
        # 3. Lateral velocity penalty
        r_lat = - 2.0 * (vy ** 2)
        
        # 4. Energy consumption penalty
        r_tau = - 0.0003 * np.sum(tau ** 2)
        
        # 5. Alive bonus
        r_alive = 1.0
        
        reward = r_vel + r_ori + r_height + r_lat + r_tau + r_alive
        
        # Termination condition
        done = False
        if height < 0.17 or height > 0.45 or abs(roll) > 0.65 or abs(pitch) > 0.65:
            done = True
            reward -= 5.0  # Fall penalty
            
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
