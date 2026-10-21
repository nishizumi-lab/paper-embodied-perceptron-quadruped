import os
import sys
import time
import numpy as np
import multiprocessing as mp

from quadruped_env import QuadrupedEnv
from models import ProposedVM, MinimalPD, BaselineMLP

def evaluate_candidate(args):
    model_type, params, seed, max_steps, lambda_l1 = args
    env = QuadrupedEnv(max_steps=max_steps)
    np.random.seed(seed)
    
    if model_type == "proposed":
        model = ProposedVM(seed=seed)
    elif model_type == "minimal_pd":
        model = MinimalPD(seed=seed)
    elif model_type == "baseline_mlp":
        model = BaselineMLP(seed=seed)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    model.set_params(params)
    
    obs = env.reset(randomize=True)
    total_reward = 0.0
    total_vx = 0.0
    steps = 0
    done = False
    
    while not done:
        tau = model.compute_torque(obs)
        obs, r, done, info = env.step(tau)
        total_reward += r
        total_vx += info["vx"]
        steps += 1
        
    avg_vx = total_vx / max(1, steps)
    
    # L1 regularization for ProposedVM R matrix
    l1_pen = 0.0
    if model_type == "proposed" and lambda_l1 > 0:
        l1_pen = lambda_l1 * np.sum(np.abs(model.R))
        
    fitness = total_reward - l1_pen
    return fitness, total_reward, avg_vx, steps

class ParallelEvolutionStrategy:
    def __init__(self, model_type="proposed", pop_size=32, sigma=0.08, lr=0.05, lambda_l1=0.05, num_workers=None):
        self.model_type = model_type
        self.pop_size = pop_size  # Number of perturbation pairs (total 2 * pop_size evaluations)
        self.sigma = sigma
        self.lr = lr
        self.lambda_l1 = lambda_l1
        self.num_workers = num_workers if num_workers is not None else min(8, mp.cpu_count())
        
        if model_type == "proposed":
            self.base_model = ProposedVM()
        elif model_type == "minimal_pd":
            self.base_model = MinimalPD()
        elif model_type == "baseline_mlp":
            self.base_model = BaselineMLP()
            
        self.theta = self.base_model.get_params().copy()
        self.dim = len(self.theta)
        
    def train(self, num_iterations=100, max_steps=300, eval_freq=5, verbose=True):
        history = {
            "iteration": [],
            "total_steps": [],
            "mean_reward": [],
            "best_reward": [],
            "avg_vx": []
        }
        
        cumulative_steps = 0
        pool = mp.Pool(processes=self.num_workers)
        
        try:
            for it in range(1, num_iterations + 1):
                t0 = time.time()
                # Generate antithetic perturbations: epsilon_i and -epsilon_i
                epsilons = np.random.randn(self.pop_size, self.dim)
                
                tasks = []
                base_seed = it * 1000
                for i in range(self.pop_size):
                    cand_pos = self.theta + self.sigma * epsilons[i]
                    cand_neg = self.theta - self.sigma * epsilons[i]
                    tasks.append((self.model_type, cand_pos, base_seed + i, max_steps, self.lambda_l1))
                    tasks.append((self.model_type, cand_neg, base_seed + i, max_steps, self.lambda_l1))
                    
                results = pool.map(evaluate_candidate, tasks)
                
                fit_pos = np.array([results[2*i][0] for i in range(self.pop_size)])
                fit_neg = np.array([results[2*i+1][0] for i in range(self.pop_size)])
                
                rew_pos = np.array([results[2*i][1] for i in range(self.pop_size)])
                rew_neg = np.array([results[2*i+1][1] for i in range(self.pop_size)])
                
                vx_all = np.array([r[2] for r in results])
                steps_all = sum(r[3] for r in results)
                cumulative_steps += steps_all
                
                # Fitness differences for gradient estimation
                fit_diff = fit_pos - fit_neg
                std_diff = np.std(fit_diff)
                if std_diff > 1e-6:
                    norm_diff = (fit_diff - np.mean(fit_diff)) / std_diff
                else:
                    norm_diff = fit_diff
                    
                # Gradient step
                grad = np.dot(epsilons.T, norm_diff) / (2.0 * self.pop_size)
                self.theta += self.lr * grad
                
                # Soft-thresholding shrinkage on R for ProposedVM to encourage exact zeros
                if self.model_type == "proposed" and self.lambda_l1 > 0:
                    n_R = self.base_model.m * self.base_model.n
                    R_vec = self.theta[:n_R]
                    shrinkage = self.lr * self.lambda_l1 * 0.05
                    self.theta[:n_R] = np.sign(R_vec) * np.maximum(0.0, np.abs(R_vec) - shrinkage)
                    
                dt = time.time() - t0
                
                mean_rew = np.mean(np.concatenate([rew_pos, rew_neg]))
                best_rew = np.max(np.concatenate([rew_pos, rew_neg]))
                mean_vx = np.mean(vx_all)
                
                history["iteration"].append(it)
                history["total_steps"].append(cumulative_steps)
                history["mean_reward"].append(float(mean_rew))
                history["best_reward"].append(float(best_rew))
                history["avg_vx"].append(float(mean_vx))
                
                if verbose and (it % eval_freq == 0 or it == 1 or it == num_iterations):
                    print(f"[{self.model_type}] Iter {it:3d}/{num_iterations} | "
                          f"Steps: {cumulative_steps:8d} | Best Rew: {best_rew:7.1f} | "
                          f"Mean Rew: {mean_rew:7.1f} | Avg vx: {mean_vx:5.2f} m/s | Time: {dt:4.1f}s")
        finally:
            pool.close()
            pool.join()
            
        self.base_model.set_params(self.theta)
        return self.base_model, history
