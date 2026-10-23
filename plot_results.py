import os
import json
import matplotlib.pyplot as plt
import numpy as np

def generate_plots():
    base_dir = os.path.dirname(__file__)
    results_path = os.path.join(base_dir, "data", "experimental_results.json")
    if not os.path.exists(results_path):
        results_path = os.path.join(base_dir, "experimental_results.json")
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return
        
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    out_dir = os.path.join(base_dir, "figures")
    if not os.path.exists(out_dir):
        out_dir = base_dir
    
    # 1. Plot Learning Curves
    exp1 = data["experiment_1_learning"]
    plt.figure(figsize=(9, 5))
    
    steps_p = exp1["proposed"]["history"]["total_steps"]
    rew_p = exp1["proposed"]["history"]["mean_reward"]
    
    steps_m = exp1["minimal_pd"]["history"]["total_steps"]
    rew_m = exp1["minimal_pd"]["history"]["mean_reward"]
    
    steps_b = exp1["baseline_mlp"]["history"]["total_steps"]
    rew_b = exp1["baseline_mlp"]["history"]["mean_reward"]
    
    plt.plot(steps_p, rew_p, label="Proposed-VM (800 params)", color="#1f77b4", linewidth=2.5)
    plt.plot(steps_m, rew_m, label="Minimal-PD (156 params)", color="#2ca02c", linewidth=2.0, linestyle="-.")
    plt.plot(steps_b, rew_b, label="Baseline-MLP (7.2k params)", color="#ff7f0e", linewidth=1.8, linestyle="--")
    
    plt.xlabel("Environment Steps", fontsize=12)
    plt.ylabel("Mean Episode Reward", fontsize=12)
    plt.title("Sample Efficiency and Learning Curves", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    p1 = os.path.join(out_dir, "fig1_learning_curves.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved: {p1}")
    
    # 2. Plot Push Recovery
    exp2 = data["experiment_2_push_recovery"]
    plt.figure(figsize=(8, 5))
    forces = exp2["forces"]
    sr_p = [r * 100 for r in exp2["success_rate_proposed"]]
    sr_b = [r * 100 for r in exp2["success_rate_mlp"]]
    sr_m = [r * 100 for r in exp2["success_rate_min"]]
    
    plt.plot(forces, sr_p, "o-", label="Proposed-VM", color="#1f77b4", linewidth=2.5)
    plt.plot(forces, sr_b, "s--", label="Baseline-MLP", color="#ff7f0e", linewidth=2.0)
    plt.plot(forces, sr_m, "^:", label="Minimal-PD", color="#7f7f7f", linewidth=1.8)
    
    plt.axhline(50, color="red", linestyle=":", alpha=0.7, label="Critical Threshold (50%)")
    plt.xlabel("Lateral Impulse Force [N] (0.1s duration)", fontsize=12)
    plt.ylabel("Gait Recovery Success Rate [%]", fontsize=12)
    plt.title("Push Recovery Performance Under Lateral Disturbance", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "fig2_push_recovery.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved: {p2}")
    
    # 3. Plot Frequency Robustness
    exp3 = data["experiment_3_frequency_robustness"]
    plt.figure(figsize=(8, 5))
    freqs = exp3["frequencies"]
    sr_p_f = [item["success_rate"] * 100 for item in exp3["proposed"]]
    sr_b_f = [item["success_rate"] * 100 for item in exp3["baseline_mlp"]]
    
    plt.plot(freqs, sr_p_f, "o-", label="Proposed-VM (f_active downsampled)", color="#1f77b4", linewidth=2.5)
    if "minimal_pd" in exp3:
        sr_m_f = [item["success_rate"] * 100 for item in exp3["minimal_pd"]]
        plt.plot(freqs, sr_m_f, "^-.", label="Minimal-PD (target downsampled)", color="#2ca02c", linewidth=2.0)
    plt.plot(freqs, sr_b_f, "s--", label="Baseline-MLP (target angle downsampled)", color="#ff7f0e", linewidth=1.8)
    
    plt.xlabel("Upper Controller Update Frequency [Hz]", fontsize=12)
    plt.ylabel("Gait Continuation Success Rate [%]", fontsize=12)
    plt.title("Robustness Against Low-Frequency Upper Control Loop", fontsize=14)
    plt.gca().invert_xaxis()  # 125 Hz -> 20 Hz
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    p3 = os.path.join(out_dir, "fig3_frequency_robustness.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Saved: {p3}")
    
    # 4. Plot Moment Arm Heatmap
    exp4 = data["experiment_4_matrix_analysis"]
    R = np.array(exp4["R_matrix"])
    plt.figure(figsize=(9, 10))
    im = plt.imshow(R, cmap="coolwarm", aspect="auto", vmin=-1.2, vmax=1.2)
    plt.colorbar(im, fraction=0.03, pad=0.04, label="Moment Arm Weight R_ij")
    
    joint_labels = [
        "FL_AA", "FL_FE", "FL_KN",
        "FR_AA", "FR_FE", "FR_KN",
        "RL_AA", "RL_FE", "RL_KN",
        "RR_AA", "RR_FE", "RR_KN"
    ]
    plt.xticks(ticks=range(12), labels=joint_labels, rotation=45, ha="right", fontsize=10)
    plt.yticks(ticks=range(32), labels=[f"M_{i+1}" for i in range(32)], fontsize=8)
    plt.xlabel("Joint Coordinates (12 DOFs)", fontsize=12)
    plt.ylabel("Virtual Muscle Index (32 Muscles, 8 per Leg)", fontsize=12)
    plt.title("Learned Moment Arm Matrix R (Sparsity & Bi-articular Structure)", fontsize=13)
    plt.tight_layout()
    p4 = os.path.join(out_dir, "fig4_moment_arm_heatmap.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"Saved: {p4}")

if __name__ == "__main__":
    generate_plots()
