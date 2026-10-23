import os
import sys
import time
import json
import numpy as np

from quadruped_env import QuadrupedEnv
from models import ProposedVM, MinimalPD, BaselineMLP
from train import ParallelEvolutionStrategy

def run_experiment_1_learning(num_iterations=40, pop_size=16, num_workers=4):
    print("=" * 60)
    print("EXPERIMENT 1: Learning Curve & Sample Efficiency")
    print("=" * 60)
    
    results = {}
    
    # 1. Proposed-VM
    print("\n--- Training Proposed-VM (Virtual Musculoskeletal Layer) ---")
    es_prop = ParallelEvolutionStrategy(model_type="proposed", pop_size=pop_size, sigma=0.04, lr=0.04, lambda_l1=0.005, num_workers=num_workers)
    model_prop, hist_prop = es_prop.train(num_iterations=num_iterations, max_steps=300, eval_freq=5)
    results["proposed"] = {
        "history": hist_prop,
        "best_params": model_prop.get_params().tolist(),
        "final_mean_reward": float(hist_prop["mean_reward"][-1]),
        "final_best_reward": float(hist_prop["best_reward"][-1]),
        "final_vx": float(hist_prop["avg_vx"][-1]),
        "total_steps": int(hist_prop["total_steps"][-1])
    }
    
    # 2. Minimal-PD (Ablation)
    print("\n--- Training Minimal-PD (Ablation: Independent Joint PD) ---")
    es_min = ParallelEvolutionStrategy(model_type="minimal_pd", pop_size=pop_size, sigma=0.04, lr=0.04, lambda_l1=0.0, num_workers=num_workers)
    model_min, hist_min = es_min.train(num_iterations=num_iterations, max_steps=300, eval_freq=5)
    results["minimal_pd"] = {
        "history": hist_min,
        "best_params": model_min.get_params().tolist(),
        "final_mean_reward": float(hist_min["mean_reward"][-1]),
        "final_best_reward": float(hist_min["best_reward"][-1]),
        "final_vx": float(hist_min["avg_vx"][-1]),
        "total_steps": int(hist_min["total_steps"][-1])
    }
    
    # 3. Baseline-MLP (Conventional)
    print("\n--- Training Baseline-MLP (Conventional Deep MLP + PD) ---")
    es_mlp = ParallelEvolutionStrategy(model_type="baseline_mlp", pop_size=pop_size, sigma=0.04, lr=0.03, lambda_l1=0.0, num_workers=num_workers)
    model_mlp, hist_mlp = es_mlp.train(num_iterations=num_iterations, max_steps=300, eval_freq=5)
    results["baseline_mlp"] = {
        "history": hist_mlp,
        "best_params": model_mlp.get_params().tolist(),
        "final_mean_reward": float(hist_mlp["mean_reward"][-1]),
        "final_best_reward": float(hist_mlp["best_reward"][-1]),
        "final_vx": float(hist_mlp["avg_vx"][-1]),
        "total_steps": int(hist_mlp["total_steps"][-1])
    }
    
    return results, model_prop, model_mlp, model_min

def run_experiment_2_push_recovery(model_prop, model_mlp, model_min, num_trials=10):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Lateral Push Recovery (Disturbance Rejection)")
    print("=" * 60)
    
    # Lateral impulse forces to test (in Newtons) from 20 N to 180 N
    forces = np.linspace(20.0, 180.0, 9)
    
    def test_model_push(model, force_y):
        success_count = 0
        env = QuadrupedEnv(max_steps=300)
        for trial in range(num_trials):
            obs = env.reset(randomize=False)
            done = False
            # Push at step 80 (approx 0.64s after start, when walking is steady)
            pushed = False
            while not done:
                if env.step_count == 80 and not pushed:
                    env.apply_lateral_impulse(force_y, duration_sec=0.1)
                    pushed = True
                tau = model.compute_torque(obs)
                obs, r, done, info = env.step(tau)
            if env.step_count >= 250 and info["height"] > 0.22:
                success_count += 1
        return success_count / num_trials
        
    rates_prop = []
    rates_mlp = []
    rates_min = []
    
    for f in forces:
        sr_p = test_model_push(model_prop, f)
        sr_m = test_model_push(model_mlp, f)
        sr_min = test_model_push(model_min, f)
        rates_prop.append(sr_p)
        rates_mlp.append(sr_m)
        rates_min.append(sr_min)
        print(f"Force: {f:5.1f} N | Proposed-VM: {sr_p*100:5.1f}% | Baseline-MLP: {sr_m*100:5.1f}% | Minimal-PD: {sr_min*100:5.1f}%")
        
    # Critical impulse force (highest force with >= 50% recovery)
    crit_prop = 0.0
    crit_mlp = 0.0
    crit_min = 0.0
    for f, r in zip(forces, rates_prop):
        if r >= 0.5: crit_prop = f
    for f, r in zip(forces, rates_mlp):
        if r >= 0.5: crit_mlp = f
    for f, r in zip(forces, rates_min):
        if r >= 0.5: crit_min = f
        
    print(f"\nCritical Recovery Force (>=50% success): Proposed={crit_prop:.1f}N, Baseline={crit_mlp:.1f}N, Minimal={crit_min:.1f}N")
    
    return {
        "forces": forces.tolist(),
        "success_rate_proposed": rates_prop,
        "success_rate_mlp": rates_mlp,
        "success_rate_min": rates_min,
        "crit_force_proposed": float(crit_prop),
        "crit_force_mlp": float(crit_mlp),
        "crit_force_min": float(crit_min)
    }

def run_experiment_3_frequency_robustness(model_prop, model_mlp, model_min, num_trials=10):
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Control Frequency Robustness (125 Hz down to 20.8 Hz)")
    print("=" * 60)
    
    # Subsampling factors for upper neural controller update:
    # 1: 125 Hz (dt_ctrl = 0.008s)
    # 2: 62.5 Hz (0.016s)
    # 4: 31.25 Hz (0.032s)
    # 6: 20.83 Hz (~20 Hz)
    update_intervals = [1, 2, 4, 6]
    freqs = [125.0 / k for k in update_intervals]
    
    def test_frequency(model, interval, model_type="proposed"):
        success_count = 0
        avg_vx_list = []
        env = QuadrupedEnv(max_steps=300)
        
        for trial in range(num_trials):
            obs = env.reset(randomize=False)
            done = False
            last_cmd = None
            
            while not done:
                if model_type == "proposed":
                    # Upper neural controller (f_active) runs at low frequency,
                    # but lower virtual musculoskeletal layer responds continuously at 125 Hz
                    if env.step_count % interval == 0 or last_cmd is None:
                        logits = model.W_neural @ obs["x_osc"] + model.W_fb @ obs["s_imu"]
                        last_cmd = model.F_max / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
                        
                    K = np.exp(np.clip(model.p_K, 1.5, 5.5))
                    D = np.exp(np.clip(model.p_D, -1.0, 2.5))
                    strain = model.R @ obs["q_joints"] + model.b
                    strain_rate = model.R @ obs["dq_joints"]
                    tension = np.maximum(0.0, K * strain + D * strain_rate + last_cmd)
                    tau = - model.R.T @ tension
                elif model_type == "minimal_pd":
                    if env.step_count % interval == 0 or last_cmd is None:
                        last_cmd = model.A_max * np.tanh(model.W_neural @ obs["x_osc"] + model.W_fb @ obs["s_imu"] + model.b)
                    tau = model.Kp * (last_cmd - obs["q_joints"]) - model.Kd * obs["dq_joints"]
                else:
                    if env.step_count % interval == 0 or last_cmd is None:
                        x = np.concatenate([obs["x_osc"], obs["s_imu"], obs["q_joints"], obs["dq_joints"]])
                        h1 = np.tanh(model.W1 @ x + model.b1)
                        h2 = np.tanh(model.W2 @ h1 + model.b2)
                        last_cmd = model.A_max * np.tanh(model.W3 @ h2 + model.b3)
                    tau = model.Kp * (last_cmd - obs["q_joints"]) - model.Kd * obs["dq_joints"]
                    
                obs, r, done, info = env.step(tau)
                
            if env.step_count >= 250 and info["height"] > 0.22:
                success_count += 1
                avg_vx_list.append(info["vx"])
                
        succ_rate = success_count / num_trials
        mean_vx = float(np.mean(avg_vx_list)) if avg_vx_list else 0.0
        return succ_rate, mean_vx

    res_prop = []
    res_mlp = []
    res_min = []
    for intv, freq in zip(update_intervals, freqs):
        sr_p, vx_p = test_frequency(model_prop, intv, model_type="proposed")
        sr_m, vx_m = test_frequency(model_mlp, intv, model_type="baseline_mlp")
        sr_min, vx_min = test_frequency(model_min, intv, model_type="minimal_pd")
        res_prop.append({"freq": freq, "success_rate": sr_p, "avg_vx": vx_p})
        res_mlp.append({"freq": freq, "success_rate": sr_m, "avg_vx": vx_m})
        res_min.append({"freq": freq, "success_rate": sr_min, "avg_vx": vx_min})
        print(f"Freq: {freq:5.1f} Hz | Proposed: {sr_p*100:5.1f}% (vx={vx_p:.2f}) | Minimal-PD: {sr_min*100:5.1f}% (vx={vx_min:.2f}) | MLP: {sr_m*100:5.1f}% (vx={vx_m:.2f})")
        
    return {
        "frequencies": freqs,
        "proposed": res_prop,
        "baseline_mlp": res_mlp,
        "minimal_pd": res_min
    }

def run_experiment_4_matrix_analysis(model_prop):
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Structural Analysis of Learned Moment Arm Matrix R")
    print("=" * 60)
    
    R = model_prop.R.copy()  # Shape: (32, 12)
    threshold = 0.05
    
    total_elements = R.size
    zero_elements = np.sum(np.abs(R) < threshold)
    sparsity_ratio = zero_elements / total_elements
    
    # Classify muscles for each leg (8 muscles per leg)
    mono_articular_count = 0
    bi_articular_count = 0
    other_count = 0
    
    for leg in range(4):
        m_start = leg * 8
        j_start = leg * 3
        for m_idx in range(m_start, m_start + 8):
            leg_weights = np.abs(R[m_idx, j_start : j_start + 3])
            active_joints = np.where(leg_weights > threshold)[0]
            if len(active_joints) == 1:
                mono_articular_count += 1
            elif len(active_joints) == 2 and (1 in active_joints and 2 in active_joints):
                bi_articular_count += 1
            else:
                other_count += 1
                
    total_muscles = 32
    mono_ratio = mono_articular_count / total_muscles
    bi_ratio = bi_articular_count / total_muscles
    
    print(f"Total R Matrix Elements: {total_elements}")
    print(f"Near-Zero Elements (< {threshold}): {zero_elements} ({sparsity_ratio*100:.1f}%)")
    print(f"Mono-articular Muscles: {mono_articular_count}/{total_muscles} ({mono_ratio*100:.1f}%)")
    print(f"Bi-articular Muscles (Hip-Knee coupled): {bi_articular_count}/{total_muscles} ({bi_ratio*100:.1f}%)")
    print(f"Other / Unclassified: {other_count}/{total_muscles}")
    
    return {
        "R_matrix": R.tolist(),
        "total_elements": int(total_elements),
        "zero_elements": int(zero_elements),
        "sparsity_ratio": float(sparsity_ratio),
        "mono_articular_count": int(mono_articular_count),
        "bi_articular_count": int(bi_articular_count),
        "mono_ratio": float(mono_ratio),
        "bi_ratio": float(bi_ratio)
    }

if __name__ == "__main__":
    t_start = time.time()
    
    exp1_res, m_prop, m_mlp, m_min = run_experiment_1_learning(num_iterations=40, pop_size=16, num_workers=4)
    exp2_res = run_experiment_2_push_recovery(m_prop, m_mlp, m_min, num_trials=10)
    exp3_res = run_experiment_3_frequency_robustness(m_prop, m_mlp, m_min, num_trials=10)
    exp4_res = run_experiment_4_matrix_analysis(m_prop)
    
    all_data = {
        "experiment_1_learning": exp1_res,
        "experiment_2_push_recovery": exp2_res,
        "experiment_3_frequency_robustness": exp3_res,
        "experiment_4_matrix_analysis": exp4_res,
        "execution_time_seconds": time.time() - t_start
    }
    
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "experimental_results.json")
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
        
    print("\n" + "=" * 60)
    print(f"ALL EXPERIMENTS COMPLETED in {time.time() - t_start:.1f}s!")
    print(f"Saved results to: {out_path}")
    print("=" * 60)
