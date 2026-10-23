import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

def relu(x):
    return np.maximum(0.0, x)

class ProposedVM:
    """
    Proposed Embodied Perceptron Virtual Musculoskeletal Layer + Minimal Neural Controller.
    Strictly non-negative muscle tension formulation:
      strain = R @ q + b
      strain_rate = R @ dq
      f_active = F_max * sigmoid(W_neural @ x_osc + W_fb @ s_imu)
      tension = relu(K * strain + D * strain_rate + f_active)  # >= 0 always!
      tau = - R.T @ tension
    Total parameters = 32*12 (R) + 32 (b) + 32 (K) + 32 (D) + 32*4 (W_neural) + 32*6 (W_fb) = 800.
    """
    def __init__(self, seed=42):
        self.m = 32  # 32 virtual muscles (8 per leg)
        self.n = 12  # 12 joints
        
        # 1. Moment arm matrix R (32 x 12)
        # Sign convention for tau = - R.T @ f (f >= 0 tension):
        # Desired tau > 0 -> -R.T > 0 -> R = -1
        # Desired tau < 0 -> -R.T < 0 -> R = +1
        R_init = np.zeros((self.m, self.n))
        for leg in range(4):
            m_s = leg * 8
            j_s = leg * 3
            # Muscle 0: Hip Abductor (tau > 0 -> R = -1.0)
            R_init[m_s + 0, j_s + 0] = -1.0
            # Muscle 1: Hip Adductor (tau < 0 -> R = 1.0)
            R_init[m_s + 1, j_s + 0] = 1.0
            # Muscle 2: Hip Flexor (swing forward, tau < 0 -> R = 1.0)
            R_init[m_s + 2, j_s + 1] = 1.0
            # Muscle 3: Hip Extensor (push backward, tau > 0 -> R = -1.0)
            R_init[m_s + 3, j_s + 1] = -1.0
            # Muscle 4: Knee Flexor (knee flexion / lift, tau > 0 -> R = -1.0)
            R_init[m_s + 4, j_s + 2] = -1.0
            # Muscle 5: Knee Extensor (knee extension, tau < 0 -> R = 1.0)
            R_init[m_s + 5, j_s + 2] = 1.0
            # Muscle 6: Bi-articular Rectus Femoris-like (Hip flex tau < 0 / Knee ext tau < 0)
            R_init[m_s + 6, j_s + 1] = 0.7
            R_init[m_s + 6, j_s + 2] = 0.7
            # Muscle 7: Bi-articular Hamstring-like (Hip ext tau > 0 / Knee flex tau > 0)
            R_init[m_s + 7, j_s + 1] = -0.7
            R_init[m_s + 7, j_s + 2] = -0.7
            
        self.R = R_init.copy()
        
        # 2. Pretension bias b (32)
        self.b = np.zeros(self.m)
        
        # 3. Log-stiffness p_K and Log-damping p_D (32 each)
        # K ~ 28 N/rad, D ~ 1.8 Ns/rad
        self.p_K = np.full(self.m, np.log(28.0))
        self.p_D = np.full(self.m, np.log(1.8))
        
        # 4. Neural controller weights
        np.random.seed(seed)
        self.W_neural = np.random.randn(self.m, 4) * 0.05
        self.W_fb = np.random.randn(self.m, 6) * 0.02
        
        # Trot pattern initialization for diagonal coupling with reciprocal inhibition
        # Diagonal pair 1: FL (leg 0), RR (leg 3)
        # Diagonal pair 2: FR (leg 1), RL (leg 2)
        w_sw = 2.4
        w_st = 2.4
        for leg in [0, 3]:
            m_s = leg * 8
            # Swing on sin(psi): Hip flex, Knee ext, Bi-RF
            self.W_neural[m_s + 2, 0] = w_sw
            self.W_neural[m_s + 5, 0] = w_sw * 0.7
            self.W_neural[m_s + 6, 0] = w_sw * 1.0
            self.W_neural[m_s + 3, 0] = -w_sw * 0.8
            self.W_neural[m_s + 4, 0] = -w_sw * 0.8
            # Stance on cos(psi): Hip ext, Knee flex, Bi-Ham
            self.W_neural[m_s + 3, 1] = w_st
            self.W_neural[m_s + 4, 1] = w_st * 0.7
            self.W_neural[m_s + 7, 1] = w_st * 1.0
            self.W_neural[m_s + 2, 1] = -w_st * 0.8
            self.W_neural[m_s + 5, 1] = -w_st * 0.8
            
        for leg in [1, 2]:
            m_s = leg * 8
            self.W_neural[m_s + 2, 2] = w_sw
            self.W_neural[m_s + 5, 2] = w_sw * 0.7
            self.W_neural[m_s + 6, 2] = w_sw * 1.0
            self.W_neural[m_s + 3, 2] = -w_sw * 0.8
            self.W_neural[m_s + 4, 2] = -w_sw * 0.8
            
            self.W_neural[m_s + 3, 3] = w_st
            self.W_neural[m_s + 4, 3] = w_st * 0.7
            self.W_neural[m_s + 7, 3] = w_st * 1.0
            self.W_neural[m_s + 2, 3] = -w_st * 0.8
            self.W_neural[m_s + 5, 3] = -w_st * 0.8
            
        # Posture feedback
        for leg in range(4):
            m_s = leg * 8
            is_left = (leg in [0, 2])
            sign = 1.0 if is_left else -1.0
            self.W_fb[m_s + 0, 0] = -0.3 * sign  # Hip Abd on roll
            self.W_fb[m_s + 1, 0] = 0.3 * sign   # Hip Add on roll
            
        self.F_max = 24.0  # Maximum muscle active force [N]

    def get_params(self):
        return np.concatenate([
            self.R.flatten(),
            self.b,
            self.p_K,
            self.p_D,
            self.W_neural.flatten(),
            self.W_fb.flatten()
        ])

    def set_params(self, params):
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

    def compute_torque(self, obs):
        x_osc = obs["x_osc"]
        s_imu = obs["s_imu"]
        q = obs["q_joints"]
        dq = obs["dq_joints"]
        
        # Upper minimal neural controller: active muscle contraction
        logits = self.W_neural @ x_osc + self.W_fb @ s_imu
        f_active = self.F_max * sigmoid(logits)
        
        # Lower virtual musculoskeletal layer:
        # Strictly non-negative tendon tension
        K = np.exp(np.clip(self.p_K, 1.5, 5.5))
        D = np.exp(np.clip(self.p_D, -1.0, 2.5))
        
        muscle_strain = self.R @ q + self.b
        muscle_strain_rate = self.R @ dq
        
        total_muscle_tension = relu(K * muscle_strain + D * muscle_strain_rate + f_active)
        tau = - self.R.T @ total_muscle_tension
        return tau


class MinimalPD:
    """
    Ablation Baseline: Minimal Neural Controller + Independent Linear PD (R = I fixed).
    Fair comparison: receives same trot oscillator input and comparable initial trot bias.
    Parameters: W_neural (12x4=48) + W_fb (12x6=72) + b (12) + Kp (12) + Kd (12) = 156 params.
    """
    def __init__(self, seed=42):
        self.n = 12
        np.random.seed(seed)
        self.W_neural = np.random.randn(self.n, 4) * 0.05
        self.W_fb = np.random.randn(self.n, 6) * 0.02
        self.b = np.zeros(self.n)
        self.Kp = np.full(self.n, 35.0)
        self.Kd = np.full(self.n, 1.8)
        self.A_max = 0.35  # Max joint angle displacement (rad)
        
        # Fair initial trot bias matching proposed controller:
        # Leg 0 (FL) & Leg 3 (RR): swing on sin(psi), stance on cos(psi)
        # Hip FE (joint 1): flex (<0) on sin, ext (>0) on cos
        # Knee FE (joint 2): ext (<0) on sin, flex (>0) on cos
        for leg in [0, 3]:
            j_s = leg * 3
            self.W_neural[j_s + 1, 0] = -1.2  # Hip flex
            self.W_neural[j_s + 1, 1] = 1.2   # Hip ext
            self.W_neural[j_s + 2, 0] = -1.0  # Knee ext
            self.W_neural[j_s + 2, 1] = 1.0   # Knee flex
            
        for leg in [1, 2]:
            j_s = leg * 3
            self.W_neural[j_s + 1, 2] = -1.2
            self.W_neural[j_s + 1, 3] = 1.2
            self.W_neural[j_s + 2, 2] = -1.0
            self.W_neural[j_s + 2, 3] = 1.0

    def get_params(self):
        return np.concatenate([
            self.W_neural.flatten(),
            self.W_fb.flatten(),
            self.b,
            self.Kp,
            self.Kd
        ])

    def set_params(self, params):
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

    def compute_torque(self, obs):
        x_osc = obs["x_osc"]
        s_imu = obs["s_imu"]
        q = obs["q_joints"]
        dq = obs["dq_joints"]
        
        delta_q_target = self.A_max * np.tanh(self.W_neural @ x_osc + self.W_fb @ s_imu + self.b)
        tau = self.Kp * (delta_q_target - q) - self.Kd * dq
        return tau


class BaselineMLP:
    """
    Conventional Baseline: 3-layer MLP (Input 34 -> 64 -> 64 -> 12) + Independent PD.
    Input: x_osc (4) + s_imu (6) + q (12) + dq (12) = 34 dimensions.
    Total parameters = 7,180.
    """
    def __init__(self, seed=42):
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

    def get_params(self):
        return np.concatenate([
            self.W1.flatten(), self.b1,
            self.W2.flatten(), self.b2,
            self.W3.flatten(), self.b3
        ])

    def set_params(self, params):
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

    def compute_torque(self, obs):
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
