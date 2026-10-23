"""並列進化戦略（Parallel Evolution Strategies）による共設計最適化モジュール

本モジュールは、勾配計算を行わずにゼロ次最適化によって高次元パラメータ空間を探索する
並列進化戦略（ES）エンジンを提供し、仮想筋骨格レイヤーの力学パラメータと
上位極小神経回路の重みを進化強化学習によって同時に最適化する。
"""

import os
import sys
import time
import numpy as np
import multiprocessing as mp

from quadruped_env import QuadrupedEnv
from models import ProposedVM, MinimalPD, BaselineMLP


def evaluate_candidate(args: tuple) -> tuple[float, float, float, int]:
    """並列ワーカー内で実行される単一パラメータ個体の評価関数

    Args:
        args (tuple): 以下の要素からなるタプル。
            - model_type (str): モデル識別子 ("proposed", "minimal_pd", "baseline_mlp")。
            - params (np.ndarray): 評価対象の平坦化パラメータベクトル。
            - seed (int): 環境乱数シード。
            - max_steps (int): 評価エピソードの最大ステップ数。
            - lambda_l1 (float): モーメントアーム行列に対する L1 正則化係数。

    Returns:
        tuple[float, float, float, int]:
            - fitness (float): 正則化ペナルティを差し引いた適応度（最適化指標）。
            - total_reward (float): 環境から得られた純粋な累積報酬。
            - avg_vx (float): エピソード内の平均前進速度 [m/s]。
            - steps (int): 転倒せずに生存した制御ステップ数。
    """
    model_type, params, seed, max_steps, lambda_l1 = args
    env = QuadrupedEnv(max_steps=max_steps)
    np.random.seed(seed)

    # モデルインスタンスの生成とパラメータ展開
    if model_type == "proposed":
        model = ProposedVM(seed=seed)
    elif model_type == "minimal_pd":
        model = MinimalPD(seed=seed)
    elif model_type == "baseline_mlp":
        model = BaselineMLP(seed=seed)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.set_params(params)

    # ロールアウトシミュレーションの実行
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

    # 提案手法に対するモーメントアーム行列 R の L1 正則化ペナルティ
    l1_pen = 0.0
    if model_type == "proposed" and lambda_l1 > 0:
        l1_pen = lambda_l1 * np.sum(np.abs(model.R))

    fitness = total_reward - l1_pen
    return fitness, total_reward, avg_vx, steps


class ParallelEvolutionStrategy:
    """対称摂動を用いる並列進化戦略（Natural Evolution Strategies 変種）

    数理原理:
        パラメータ平均ベクトル theta に対し、母集団 P 組の対称標準正規摂動
        theta + sigma * epsilon_i および theta - sigma * epsilon_i を生成する。
        各個体の適応度評価値 J(+), J(-) の差分から、パラメータ勾配の不偏推定量
        grad = sum( (J(+) - J(-)) * epsilon_i ) / (2 * P * sigma)
        を計算し、確率的勾配上昇法によって theta を更新する。
        さらに、モーメントアーム行列 R に対してソフト閾値収縮（Soft-Thresholding）
        を適用し、解剖学的に意味のある疎な結合構造を形成する。

    Attributes:
        model_type (str): 対象モデルの種類。
        pop_size (int): 摂動ペア数（全評価個体数は 2 * pop_size）。
        sigma (float): 探索摂動の標準偏差。
        lr (float): パラメータ更新の学習率。
        lambda_l1 (float): L1 スパース正則化強度。
        num_workers (int): 並列評価プロセス数。
        theta (np.ndarray): 最適化対象の平均パラメータベクトル。
        dim (int): パラメータ空間の次元数。
    """

    def __init__(self, model_type: str = "proposed", pop_size: int = 16, sigma: float = 0.04,
                 lr: float = 0.04, lambda_l1: float = 0.005, num_workers: int = None):
        """並列進化戦略オプティマイザの初期化

        Args:
            model_type (str): 最適化対象モデル ("proposed", "minimal_pd", "baseline_mlp")。
            pop_size (int): 摂動ペアの個数（評価回数は 2 * pop_size）。
            sigma (float): 探索ガウスノイズの標準偏差（デフォルト: 0.04）。
            lr (float): 最適化学習率（デフォルト: 0.04）。
            lambda_l1 (float): L1 スパース正則化強度（デフォルト: 0.005）。
            num_workers (int, optional): 並列プロセス数。None の場合は論理コア数に基づき自動設定。
        """
        self.model_type = model_type
        self.pop_size = pop_size
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

    def train(self, num_iterations: int = 40, max_steps: int = 300, eval_freq: int = 5, verbose: bool = True) -> tuple[object, dict]:
        """反復最適化ループの実行

        Args:
            num_iterations (int): 最適化イテレーション総数。
            max_steps (int): 各ロールアウトの最大シミュレーションステップ数。
            eval_freq (int): コンソールへの進捗表示頻度（イテレーション間隔）。
            verbose (bool): 進捗状況を標準出力に出力するか否か。

        Returns:
            tuple[object, dict]:
                - base_model: 最適化後のパラメータがロードされたモデルインスタンス。
                - history: イテレーションごとの総ステップ数、平均報酬、最高報酬、平均前進速度の履歴辞書。
        """
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
                # 対称摂動ノイズの生成: epsilon_i ~ N(0, I)
                epsilons = np.random.randn(self.pop_size, self.dim)

                tasks = []
                base_seed = it * 1000
                for i in range(self.pop_size):
                    cand_pos = self.theta + self.sigma * epsilons[i]
                    cand_neg = self.theta - self.sigma * epsilons[i]
                    tasks.append((self.model_type, cand_pos, base_seed + i, max_steps, self.lambda_l1))
                    tasks.append((self.model_type, cand_neg, base_seed + i, max_steps, self.lambda_l1))

                # 並列ロールアウト評価の実行
                results = pool.map(evaluate_candidate, tasks)

                fit_pos = np.array([results[2*i][0] for i in range(self.pop_size)])
                fit_neg = np.array([results[2*i+1][0] for i in range(self.pop_size)])

                rew_pos = np.array([results[2*i][1] for i in range(self.pop_size)])
                rew_neg = np.array([results[2*i+1][1] for i in range(self.pop_size)])

                vx_all = np.array([r[2] for r in results])
                steps_all = sum(r[3] for r in results)
                cumulative_steps += steps_all

                # 対称差分による適応度勾配推定量の算出
                fit_diff = fit_pos - fit_neg
                std_diff = np.std(fit_diff)
                if std_diff > 1e-6:
                    norm_diff = (fit_diff - np.mean(fit_diff)) / std_diff
                else:
                    norm_diff = fit_diff

                # 勾配ステップの更新: theta = theta + lr * grad
                grad = np.dot(epsilons.T, norm_diff) / (2.0 * self.pop_size)
                self.theta += self.lr * grad

                # 提案手法に対するソフト閾値収縮（Soft-Thresholding Shrinkage）による L1 スパース化
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
