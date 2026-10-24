"""論文内シミュレーション実験自動実行・測定スクリプトモジュール

本モジュールは、論文で報告されている4つの主要実験を一括実行し、
測定結果を JSON 形式（data/experimental_results.json）として保存する。

実行実験一覧:
    実験1: 学習曲線とサンプル効率の比較（Proposed-VM, Minimal-PD, Baseline-MLP）
    実験2: 横方向撃力インパルス外乱耐性試験（Push Recovery: 20 N 〜 180 N、初期姿勢・歩行位相ランダム化20試行）
    実験3: 上位制御周波数の低減に対する歩行ロバスト性試験（125 Hz 〜 20.8 Hz、初期姿勢ランダム化20試行）
    実験4: 最適化された仮想モーメントアーム行列 R の解剖学的構造・スパース性解析
"""

import os
import sys
import time
import json
import numpy as np

from quadruped_env import QuadrupedEnv
from models import ProposedVM, MinimalPD, BaselineMLP
from train import ParallelEvolutionStrategy


def run_experiment_1_learning(num_iterations: int = 40, pop_size: int = 16, num_workers: int = 4) -> tuple[dict, object, object, object]:
    """実験1: 学習曲線とサンプル効率の評価実験

    同一環境および報酬関数のもとで、提案手法（Proposed-VM）、アブレーションモデル（Minimal-PD）、
    および従来手法（Baseline-MLP）を進化強化学習により最適化し、学習曲線データを取得する。

    Args:
        num_iterations (int): 最適化イテレーション数（デフォルト: 40）。
        pop_size (int): 進化戦略の摂動ペア数（デフォルト: 16）。
        num_workers (int): 並列プロセス数（デフォルト: 4）。

    Returns:
        tuple[dict, object, object, object]:
            - results (dict): 3モデルの学習履歴および最終性能指標を格納した辞書。
            - model_prop (ProposedVM): 学習済み提案モデル。
            - model_mlp (BaselineMLP): 学習済みBaseline-MLPモデル。
            - model_min (MinimalPD): 学習済みMinimal-PDモデル。
    """
    print("=" * 60)
    print("EXPERIMENT 1: Learning Curve & Sample Efficiency")
    print("=" * 60)

    results = {}

    # 1. 提案手法（Proposed-VM）の学習
    print("\n--- Training Proposed-VM (Virtual Musculoskeletal Layer) ---")
    es_prop = ParallelEvolutionStrategy(model_type="proposed", pop_size=pop_size, sigma=0.04, lr=0.04, lambda_l1=0.08, num_workers=num_workers)
    model_prop, hist_prop = es_prop.train(num_iterations=num_iterations, max_steps=300, eval_freq=5)
    results["proposed"] = {
        "history": hist_prop,
        "best_params": model_prop.get_params().tolist(),
        "final_mean_reward": float(hist_prop["mean_reward"][-1]),
        "final_best_reward": float(hist_prop["best_reward"][-1]),
        "final_vx": float(hist_prop["avg_vx"][-1]),
        "final_det_reward": float(hist_prop["det_reward"][-1]),
        "final_det_vx": float(hist_prop["det_vx"][-1]),
        "total_steps": int(hist_prop["total_steps"][-1])
    }

    # 2. アブレーションモデル（Minimal-PD）の学習
    print("\n--- Training Minimal-PD (Ablation: Independent Joint PD) ---")
    es_min = ParallelEvolutionStrategy(model_type="minimal_pd", pop_size=pop_size, sigma=0.04, lr=0.04, lambda_l1=0.0, num_workers=num_workers)
    model_min, hist_min = es_min.train(num_iterations=num_iterations, max_steps=300, eval_freq=5)
    results["minimal_pd"] = {
        "history": hist_min,
        "best_params": model_min.get_params().tolist(),
        "final_mean_reward": float(hist_min["mean_reward"][-1]),
        "final_best_reward": float(hist_min["best_reward"][-1]),
        "final_vx": float(hist_min["avg_vx"][-1]),
        "final_det_reward": float(hist_min["det_reward"][-1]),
        "final_det_vx": float(hist_min["det_vx"][-1]),
        "total_steps": int(hist_min["total_steps"][-1])
    }

    # 3. 従来手法モデル（Baseline-MLP）の学習
    print("\n--- Training Baseline-MLP (Conventional Deep MLP + PD) ---")
    es_mlp = ParallelEvolutionStrategy(model_type="baseline_mlp", pop_size=pop_size, sigma=0.04, lr=0.03, lambda_l1=0.0, num_workers=num_workers)
    model_mlp, hist_mlp = es_mlp.train(num_iterations=num_iterations, max_steps=300, eval_freq=5)
    results["baseline_mlp"] = {
        "history": hist_mlp,
        "best_params": model_mlp.get_params().tolist(),
        "final_mean_reward": float(hist_mlp["mean_reward"][-1]),
        "final_best_reward": float(hist_mlp["best_reward"][-1]),
        "final_vx": float(hist_mlp["avg_vx"][-1]),
        "final_det_reward": float(hist_mlp["det_reward"][-1]),
        "final_det_vx": float(hist_mlp["det_vx"][-1]),
        "total_steps": int(hist_mlp["total_steps"][-1])
    }

    return results, model_prop, model_mlp, model_min


def run_experiment_2_push_recovery(model_prop: ProposedVM, model_mlp: BaselineMLP, model_min: MinimalPD, num_trials: int = 20) -> dict:
    """実験2: 横方向インパルス撃力外乱に対する姿勢回復性能評価（位相・初期状態ランダム化統計評価）

    各外力条件につき num_trials 回の独立試行を実施する。
    統計的厳密性を担保するため、試行ごとに初期関節角度にランダム摂動（randomize=True）を付与し、
    さらに外力印加タイミングを定常歩行1周期分（ステップ70〜148）から一様ランダムにサンプリングすることで、
    遊脚期・接地瞬間・両脚支持期などあらゆる歩行位相における外乱回復成功率（Success Rate）を測定する。

    Args:
        model_prop (ProposedVM): 学習済み提案モデル。
        model_mlp (BaselineMLP): 学習済みBaseline-MLPモデル。
        model_min (MinimalPD): 学習済みMinimal-PDモデル。
        num_trials (int): 各外力条件での独立試行回数（デフォルト: 20回）。

    Returns:
        dict: 各加振力における3モデルの回復成功率および限界耐性力 [N] を格納した辞書。
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Lateral Push Recovery (Disturbance Rejection)")
    print("=" * 60)

    # 評価する横方向インパルス撃力配列（20 N から 180 N、20 N 刻み）
    forces = np.linspace(20.0, 180.0, 9)

    def test_model_push(model, force_y: float) -> float:
        success_count = 0
        env = QuadrupedEnv(max_steps=300)
        for trial in range(num_trials):
            np.random.seed(2000 + trial)
            rng = np.random.default_rng(2000 + trial)
            # 1歩容周期（約78制御ステップ）全域にわたって外乱印加タイミングを一様サンプリング
            push_step = int(rng.integers(70, 148))
            # 正負両方向（左・右）の横方向外乱を交互に試験
            direction = 1.0 if (trial % 2 == 0) else -1.0

            obs = env.reset(randomize=True)
            done = False
            pushed = False
            while not done:
                if env.step_count == push_step and not pushed:
                    env.apply_lateral_impulse(direction * force_y, duration_sec=0.1)
                    pushed = True
                tau = model.compute_torque(obs)
                obs, r, done, info = env.step(tau)
            # 250ステップ以上生存し、胴体高さを維持していれば成功と判定
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

    # 成功率 50% 以上を維持できる最大外乱力を限界耐性力（Critical Recovery Force）と定義
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


def run_experiment_3_frequency_robustness(model_prop: ProposedVM, model_mlp: BaselineMLP, model_min: MinimalPD, num_trials: int = 20) -> dict:
    """実験3: 上位制御周波数の低減に対する歩行ロバスト性評価

    上位制御器の計算更新周波数を標準の 125 Hz から 62.5 Hz, 31.25 Hz, 20.83 Hz まで
    段階的に低下（間引き更新）させ、初期姿勢摂動（randomize=True）を伴う num_trials 回の試行における
    歩行継続成功率およびエピソード全体の平均前進速度を測定する。

    Args:
        model_prop (ProposedVM): 学習済み提案モデル。
        model_mlp (BaselineMLP): 学習済みBaseline-MLPモデル。
        model_min (MinimalPD): 学習済みMinimal-PDモデル。
        num_trials (int): 各周波数条件での独立試行回数（デフォルト: 20回）。

    Returns:
        dict: 各更新周波数における3モデルの成功率および平均前進速度を格納した辞書。
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Control Frequency Robustness (125 Hz down to 20.8 Hz)")
    print("=" * 60)

    # 制御間引き係数: 1 (125 Hz), 2 (62.5 Hz), 4 (31.25 Hz), 6 (20.83 Hz)
    update_intervals = [1, 2, 4, 6]
    freqs = [125.0 / k for k in update_intervals]

    def test_frequency(model, interval: int, model_type: str = "proposed") -> tuple[float, float]:
        success_count = 0
        avg_vx_list = []
        env = QuadrupedEnv(max_steps=300)

        for trial in range(num_trials):
            np.random.seed(3000 + trial)
            obs = env.reset(randomize=True)
            done = False
            last_cmd = None
            ep_vx_sum = 0.0
            ep_steps = 0

            while not done:
                if model_type == "proposed":
                    # 提案手法: 上位神経指令 f_active は低周波で更新・ホールドされるが、
                    # 下位の仮想筋骨格レイヤー（受動バネ・ダンパ特性）は 125 Hz で連続応答する
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
                ep_vx_sum += info["vx"]
                ep_steps += 1

            if env.step_count >= 250 and info["height"] > 0.22:
                success_count += 1
                avg_vx_list.append(ep_vx_sum / max(1, ep_steps))

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


def run_experiment_4_matrix_analysis(model_prop: ProposedVM) -> dict:
    """実験4: 最適化された仮想モーメントアーム行列 R の構造・解剖学的解析

    学習後の行列 R（32筋 x 12関節 = 384要素）について以下を定量評価する:
    1. 全体スパース率（384要素中の近ゼロ要素率）および同一脚内ブロック（96要素）内のスパース率。
    2. 異なる脚間の非物理的交差結合数が厳密に 0（解剖学的ブロック対角制約 100% 遵守）であることの検証。
    3. 全32本の仮想筋の完全分類（単関節筋、股関節-膝関節二関節筋、その他多関節筋、休止筋）。

    Args:
        model_prop (ProposedVM): 最適化後の提案モデル。

    Returns:
        dict: スパース率、単関節筋数、二関節筋数、交差結合数などの構造統計辞書。
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Structural Analysis of Learned Moment Arm Matrix R")
    print("=" * 60)

    R = model_prop.R.copy()  # 形状: (32, 12)
    mask = model_prop.R_mask
    threshold = 0.05

    total_elements = R.size
    zero_elements = int(np.sum(np.abs(R) < threshold))
    sparsity_ratio = zero_elements / total_elements

    # 解剖学的ブロック対角内（96要素）および他脚間（288要素）の解析
    intra_limb_elements = 96
    intra_active = int(np.sum((np.abs(R) >= threshold) & (mask == 1.0)))
    intra_zero = intra_limb_elements - intra_active
    intra_sparsity_ratio = intra_zero / intra_limb_elements
    cross_limb_active = int(np.sum((np.abs(R) >= threshold) & (mask == 0.0)))

    # 全32本の仮想筋の分類
    mono_articular_count = 0
    bi_articular_count = 0
    other_multi_count = 0
    inactive_count = 0

    for leg in range(4):
        m_start = leg * 8
        j_start = leg * 3
        for m_idx in range(m_start, m_start + 8):
            leg_weights = np.abs(R[m_idx, j_start : j_start + 3])
            active_joints = np.where(leg_weights >= threshold)[0]
            if len(active_joints) == 0:
                inactive_count += 1
            elif len(active_joints) == 1:
                mono_articular_count += 1
            elif len(active_joints) == 2 and (1 in active_joints and 2 in active_joints):
                # 股関節FE（関節1）と膝関節FE（関節2）の双方に有意な重みを持つ二関節筋
                bi_articular_count += 1
            else:
                other_multi_count += 1

    total_muscles = 32
    mono_ratio = mono_articular_count / total_muscles
    bi_ratio = bi_articular_count / total_muscles

    print(f"Total R Matrix Elements: {total_elements}")
    print(f"Near-Zero Elements (< {threshold}): {zero_elements}/{total_elements} ({sparsity_ratio*100:.1f}%)")
    print(f"Intra-limb Sparsity (within 96 anatomical elements): {intra_zero}/{intra_limb_elements} ({intra_sparsity_ratio*100:.1f}%)")
    print(f"Cross-limb Anatomical Violations: {cross_limb_active}/288")
    print(f"Mono-articular Muscles: {mono_articular_count}/{total_muscles} ({mono_ratio*100:.1f}%)")
    print(f"Bi-articular Muscles (Hip-Knee coupled): {bi_articular_count}/{total_muscles} ({bi_ratio*100:.1f}%)")
    print(f"Other Multi-articular Muscles (Hip-AA coupled): {other_multi_count}/{total_muscles} ({other_multi_count/total_muscles*100:.1f}%)")
    print(f"Pruned / Inactive Muscles: {inactive_count}/{total_muscles}")

    return {
        "R_matrix": R.tolist(),
        "total_elements": int(total_elements),
        "zero_elements": int(zero_elements),
        "sparsity_ratio": float(sparsity_ratio),
        "intra_limb_elements": int(intra_limb_elements),
        "intra_zero_elements": int(intra_zero),
        "intra_sparsity_ratio": float(intra_sparsity_ratio),
        "cross_limb_active": int(cross_limb_active),
        "mono_articular_count": int(mono_articular_count),
        "bi_articular_count": int(bi_articular_count),
        "other_multi_count": int(other_multi_count),
        "inactive_count": int(inactive_count),
        "mono_ratio": float(mono_ratio),
        "bi_ratio": float(bi_ratio)
    }


if __name__ == "__main__":
    t_start = time.time()

    exp1_res, m_prop, m_mlp, m_min = run_experiment_1_learning(num_iterations=40, pop_size=16, num_workers=4)
    exp2_res = run_experiment_2_push_recovery(m_prop, m_mlp, m_min, num_trials=20)
    exp3_res = run_experiment_3_frequency_robustness(m_prop, m_mlp, m_min, num_trials=20)
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
