# 身体化パーセプトロン表現に基づく仮想多関節筋骨格レイヤーと極小神経回路の共設計による四脚歩行制御

**Co-design of Virtual Biarticular Musculoskeletal Layer and Minimal Neural Circuit Based on Embodied Perceptron for Quadruped Locomotion**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23533255.svg)](https://doi.org/10.5281/zenodo.23533255)
[![License: MIT](https://img.shields.io/badge/Code_License-MIT-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Paper_License-CC_BY_4.0-lightgrey.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![MuJoCo 3.1+](https://img.shields.io/badge/MuJoCo-3.1+-orange.svg)](https://mujoco.org/)

本リポジトリは、身体化パーセプトロン表現を電動四脚ロボット（Unitree Go2モデル相当）の仮想インピーダンス制御として実装し、極小の神経回路と下位の仮想筋骨格レイヤーを進化強化学習によって共設計（Co-design）する研究の論文本文、シミュレーション実装コード、および実験再現スクリプトを公開するものです。

論文全文は [PAPER.md](PAPER.md) をご覧ください。

---

## 概要（Abstract）

深層強化学習を用いた四脚歩行ロボットの制御では、上位のニューラルネットワークが目標関節角度を出力し、下位の関節独立型PD制御器がトルクを生成する構成が広く採用されています。しかし、各関節が独立した単関節バネとして振る舞うため、多関節間の協調運動や衝撃吸収の計算をすべて上位の巨大なネットワークに依存しており、サンプル効率の低下やブラックボックス化による解釈性の欠如が課題となっています。

本研究では、腱駆動機構の静力学モデルと3層パーセプトロンの数理的等価性に着目した身体化パーセプトロン表現を、電動四脚ロボットの仮想インピーダンス制御則として実装する手法を提案します。提案手法では、各脚内の多関節をまたぐブロック対角構造の仮想モーメントアーム行列と腱のたるみ・短縮アンローディング特性（ReLU関数）から成る厳密な非負筋張力仮想筋骨格レイヤーを構築し、固有受容感覚による順運動学胴体高さ推定および足先接地反力（GRF）に基づく位相調整（Tegotaeフィードバック）と統合して、進化強化学習により同時に最適化します。

物理シミュレータMuJoCo上の12自由度四脚ロボット（QDDモータのトルク・速度飽和特性を考慮）を用いた検証の結果、提案手法は従来の多層パーセプトロンと比較して約9分の1のパラメータ数（800パラメータ）で目標速度に一致する安定したトロット歩行（決定論的評価速度 0.61 m/s、探索母集団平均速度 0.54 m/s、最高報酬 712.1）を迅速に獲得しました。同一の歩容バイアスを持つ関節独立PD制御（Minimal-PD: 決定論的評価速度 0.18 m/s）と比較して約3.3倍の前進推進力を達成し、歩行位相・初期姿勢をランダム化した横方向撃力外乱試験において160 Nまで50%以上の回復率（100 Nまで100%回復）を実証しました。さらに、上位制御周波数を20.8 Hzまで低減しても100%の成功率で歩行を継続（前進速度 0.59 m/s）することを確認しました。

<p align="center">
  <img src="figures/mujoco_walking.gif" alt="MuJoCo Quadruped Trotting" width="560">
  <br>
  <em>MuJoCo物理シミュレータ上で獲得された安定トロット歩行（決定論的評価速度 0.61 m/s）</em>
</p>

---

## 主な研究成果

| 評価項目 | 従来手法 (Baseline-MLP) | アブレーション (Minimal-PD) | 提案手法 (Proposed-VM) |
| --- | --- | --- | --- |
| 上位制御器 | 3層MLP (64x64) | 極小神経回路 | 極小神経回路＋GRF位相調整 |
| 下位制御則 | 関節独立PD制御 | 関節独立PD制御 | 仮想筋骨格レイヤー ($R, b, K, D$) |
| 学習パラメータ総数 | 7,180 | 156 | 800（MLPの約 1/9） |
| 最終平均報酬 / 最高報酬 | 343.8 / 356.2 | 470.4 / 502.3 | 663.4 / 712.1（ピーク 719.7） |
| 決定論的評価報酬 | 355.4 | 488.5 | 705.0 |
| 前進速度 $v_x$（決定論的 / 母集団） | -0.03 / -0.05 m/s | 0.18 / 0.17 m/s | 0.61 / 0.54 m/s（目標 0.60 に一致） |
| 限界撃力外乱耐性（20試行 $\ge 50\%$） | 180.0 N（静止直立） | 140.0 N（歩行中） | 160.0 N（高速歩行中、100 Nまで100%） |
| 低周波制御 (20.8 Hz) 成功率 | 100.0%（$v_x = -0.06$） | 100.0%（$v_x = 0.20$） | 100.0%（$v_x = 0.59$ 安定維持） |
| 行列 $R$ 全体 / 脚内スパース率 | - | - | 87.0%（334/384） / 47.9%（46/96） |
| 仮想筋分類（単関節 / 二関節 / 他） | - | - | 16本（50.0%） / 10本（31.2%） / 6本（18.8%） |

<p align="center">
  <img src="figures/fig1_learning_curves.png" width="48%">
  <img src="figures/fig2_push_recovery.png" width="48%">
</p>
<p align="center">
  <img src="figures/fig3_frequency_robustness.png" width="48%">
  <img src="figures/fig4_moment_arm_heatmap.png" width="48%">
</p>

---

## リポジトリ構成

```
paper-embodied-perceptron-quadruped/
├── PAPER.md                 # 論文本文（Markdown形式）
├── PAPER.html               # 論文本文（ブラウザ閲覧・PDF印刷用HTML形式）
├── README.md                # 本文書（概要、クイックスタート、引用案内）
├── CITATION.cff             # 引用メタデータファイル（GitHub/Zenodo共通）
├── .zenodo.json             # Zenodo自動アーカイブ用メタデータ
├── LICENSE                  # デュアルライセンス（コード: MIT, 論文/データ: CC-BY-4.0）
├── requirements.txt         # 実行依存パッケージ一覧
├── go2_model.xml            # Unitree Go2相当のMuJoCo MJCFモデル
├── quadruped_env.py         # MuJoCo四脚歩行環境のラッパー（胴体座標変換・順運動学・GRF位相調整）
├── models.py                # 提案モデル（ProposedVM）および比較モデル定義
├── train.py                 # 並列進化戦略（ES）による学習エンジン
├── experiments.py           # 論文内全実験の自動実行・計測スクリプト
├── plot_results.py          # 実験結果（JSON）からの図表プロットスクリプト
├── capture_screenshots.py   # 歩行アニメーションGIFおよびレンダリング画像生成
├── figures/                 # 論文・README掲載用の図表および画像ファイル
├── data/                    # 実験測定データ（JSON）
├── docs/                    # 原稿アーカイブおよびドキュメント
└── scripts/                 # 補助ユーティリティスクリプト
```

---

## クイックスタート（環境構築と再現手順）

### 1. 依存関係のインストール

Python 3.10 以上が推奨されます。

```bash
git clone https://github.com/nishizumi-lab/paper-embodied-perceptron-quadruped.git
cd paper-embodied-perceptron-quadruped

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 学習と実験の実行（再現）

論文に記載されている全4つのシミュレーション実験（学習曲線・外乱耐性・制御周波数耐性・行列解析）を一括実行します。測定データは `data/experimental_results.json` に保存されます。

```bash
python experiments.py
```

### 3. 図表の生成

既存または新規取得された測定データから、論文用の図1〜図4を `figures/` ディレクトリに生成します。

```bash
python plot_results.py
```

### 4. 歩行アニメーションとスクリーンショットの取得

MuJoCoレンダラーを用いて、学習済みモデルの歩行シーンの高品質レンダリング画像およびアニメーションGIFを生成します。

```bash
python capture_screenshots.py
```

---

## 引用（Citation）

本研究成果またはコードベースを利用される場合は、以下の形式で引用してください。

```bibtex
@article{nishizumi2026embodied,
  title={身体化パーセプトロン表現に基づく仮想多関節筋骨格レイヤーと極小神経回路の共設計による四脚歩行制御},
  author={Nishizumi},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.23533255},
  url={https://doi.org/10.5281/zenodo.23533255}
}
```

---

## ライセンス

本リポジトリはデュアルライセンスを採用しています。

- プログラムコード（Python / XML等）：[MIT License](LICENSE)
- 論文本文（`PAPER.md`）、図表、実験データ：[Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE)
