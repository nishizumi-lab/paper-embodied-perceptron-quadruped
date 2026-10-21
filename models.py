import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

def relu(x):
    return np.maximum(0.0, x)

class ProposedVM:
    """
    Proposed Embodied Perceptron Virtual Musculoskeletal Layer + Minimal Neural Controller.
    Total parameters = 32*12 (R) + 32 (b) + 32 (K) + 32 (D) + 32*4 (W_neural) + 32*6 (W_fb) = 800.
    """
    def __init__(self, seed=42):
        self.m = 32  # 32 virtual muscles (8 per leg)
        self.n = 12  # 12 joints
        
        # 1. Moment arm matrix R (32 x 12)
        R_init = np.zeros((self.m, self.n))
        for leg in range(4):
            m_s = leg * 8
            j_s = leg * 3
            # Muscle 0: Hip Abductor (+)
            R_init[m_s + 0, j_s + 0] = 1.0
            # Muscle 1: Hip Adductor (-)
            R_init[m_s + 1, j_s + 0] = -1.0
            # Muscle 2: Hip Flexor (swing forward, thigh < 0)
            R_init[m_s + 2, j_s + 1] = -1.0
            # Muscle 3: Hip Extensor (push backward, thigh > 0)
            R_init[m_s + 3, j_s + 1] = 1.0
            # Muscle 4: Knee Flexor (knee flexion, calf < 0)
            R_init[m_s + 4, j_s + 2] = -1.0
            # Muscle 5: Knee Extensor (knee extension, calf > 0)
            R_init[m_s + 5, j_s + 2] = 1.0
            # Muscle 6: Bi-articular Rectus Femoris-like (Hip flex - / Knee ext +)
            R_init[m_s + 6, j_s + 1] = -0.7
            R_init[m_s + 6, j_s + 2] = 0.7
            # Muscle 7: Bi-articular Hamstring-like (Hip ext + / Knee flex -)
            R_init[m_s + 7, j_s + 1] = 0.7
            R_init[m_s + 7, j_s + 2] = -0.7
            
        self.R = R_init.copy()
        
        # 2. Pretension bias b (32)
        self.b = np.full(self.m, 0.05)
        
        # 3. Log-stiffness p_K and Log-damping p_D (32 each)
        # K ~ 45 N/rad, D ~ 2.0 Ns/rad
        self.p_K = np.full(self.m, np.log(45.0))
        self.p_D = np.full(self.m, np.log(2.0))
        
        # 4. Neural controller weights
        np.random.seed(seed)
        self.W_neural = np.random.randn(self.m, 4) * 0.05
        self.W_fb = np.random.randn(self.m, 6) * 0.02
        
        # Prior trot pattern initialization for diagonal coupling
        # Diagonal pair 1: FL (leg 0), RR (leg 3)
        # Diagonal pair 2: FR (leg 1), RL (leg 2)
        for leg in [0, 3]:
            m_s = leg * 8
            self.W_neural[m_s + 2, 0] = 1.2   # hip flexor with sin(psi)
            self.W_neural[m_s + 3, 1] = 1.2   # hip extensor with cos(psi)
            self.W_neural[m_s + 4, 0] = 0.8   # knee flexor with sin(psi)
            self.W_neural[m_s + 5, 1] = 1.0   # knee extensor with cos(psi)
            self.W_neural[m_s + 6, 0] = 0.6   # bi-articular flex-ext
            self.W_neural[m_s + 7, 1] = 0.6   # bi-articular ext-flex
            
        for leg in [1, 2]:
            m_s = leg * 8
            self.W_neural[m_s + 2, 2] = 1.2   # hip flexor with sin(psi+pi)
            self.W_neural[m_s + 3, 3] = 1.2   # hip extensor with cos(psi+pi)
            self.W_neural[m_s + 4, 2] = 0.8   # knee flexor with sin(psi+pi)
            self.W_neural[m_s + 5, 3] = 1.0   # knee extensor with cos(psi+pi)
            self.W_neural[m_s + 6, 2] = 0.6
            self.W_neural[m_s + 7, 3] = 0.6
            
        self.F_max = 35.0  # Maximum muscle active force [N]

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
        # tau = -R^T * phi( K(R q + b) + D R dq ) + R^T * f_active
        K = np.exp(np.clip(self.p_K, 1.5, 5.5))
        D = np.exp(np.clip(self.p_D, -1.0, 2.5))
        
        muscle_strain = self.R @ q + self.b
        muscle_strain_rate = self.R @ dq
        
        passive_force = relu(K * muscle_strain + D * muscle_strain_rate)
        total_muscle_force = passive_force - f_active
        
        tau = - self.R.T @ total_muscle_force
        return tau


class MinimalPD:
    """
    Ablation: Minimal Neural Controller + Independent Linear PD (R = I fixed).
    Parameters: W_neural (12x4=48) + W_fb (12x6=72) + b (12) + Kp (12) + Kd (12) = 156 params (~150).
    """
    def __init__(self, seed=42):
        self.n = 12
        np.random.seed(seed)
        self.W_neural = np.random.randn(self.n, 4) * 0.1
        self.W_fb = np.random.randn(self.n, 6) * 0.05
        self.b = np.zeros(self.n)
        self.Kp = np.full(self.n, 45.0)
        self.Kd = np.full(self.n, 2.0)
        self.A_max = 0.4  # Max joint angle displacement (rad)

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
    Conventional Baseline: 3-layer MLP (Input 34 -> 128 -> 128 -> 12) + Independent PD.
    Total parameters ~ 22,540 (~20k).
    """
    def __init__(self, seed=42):
        self.dim_in = 34  # x_osc (4) + s_imu (6) + q (12) + dq (12)
        self.dim_h = 128
        self.dim_out = 12
        
        np.random.seed(seed)
        self.W1 = np.random.randn(self.dim_h, self.dim_in) * np.sqrt(2.0 / self.dim_in)
        self.b1 = np.zeros(self.dim_h)
        self.W2 = np.random.randn(self.dim_h, self.dim_h) * np.sqrt(2.0 / self.dim_h)
        self.b2 = np.zeros(self.dim_h)
        self.W3 = np.random.randn(self.dim_out, self.dim_h) * np.sqrt(2.0 / self.dim_h) * 0.1
        self.b3 = np.zeros(self.dim_out)
        
        self.Kp = 45.0
        self.Kd = 2.0
        self.A_max = 0.4

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
