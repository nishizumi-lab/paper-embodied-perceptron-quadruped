import os
import re

def build_html():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paper_md_path = os.path.join(base_dir, "PAPER.md")
    
    with open(paper_md_path, "r", encoding="utf-8") as f:
        md = f.read()
        
    html_template = r"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>身体化パーセプトロン表現に基づく仮想多関節筋骨格レイヤーと極小神経回路の共設計による四脚歩行制御</title>
  <!-- KaTeX for math rendering -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
          onload="renderMathInElement(document.body, {delimiters: [{left: '$$', right: '$$', display: true}, {left: '$', right: '$', display: false}]});"></script>
  <style>
    @page {
      size: A4;
      margin: 20mm;
    }
    body {
      font-family: "Hiragino Kaku Gothic ProN", "Yu Gothic", "Meiryo", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      line-height: 1.75;
      color: #222;
      max-width: 860px;
      margin: 0 auto;
      padding: 30px 20px;
      background: #fff;
    }
    h1 {
      font-size: 24px;
      text-align: center;
      margin-bottom: 8px;
      line-height: 1.35;
      color: #111;
    }
    .subtitle {
      font-size: 15px;
      text-align: center;
      color: #555;
      margin-bottom: 16px;
      font-weight: 500;
    }
    .author-info {
      text-align: center;
      font-size: 13.5px;
      color: #444;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid #ddd;
    }
    .abstract-box {
      background: #f8f9fa;
      border: 1px solid #e2e8f0;
      border-left: 4px solid #2b6cb0;
      padding: 18px 22px;
      border-radius: 4px;
      margin-bottom: 30px;
    }
    .abstract-title {
      font-weight: bold;
      margin-bottom: 8px;
      color: #2b6cb0;
      font-size: 15px;
    }
    h2 {
      font-size: 18px;
      border-bottom: 2px solid #2b6cb0;
      padding-bottom: 4px;
      margin-top: 36px;
      margin-bottom: 14px;
      color: #1a365d;
    }
    h3 {
      font-size: 15.5px;
      margin-top: 22px;
      margin-bottom: 10px;
      color: #2d3748;
    }
    p {
      margin-bottom: 14px;
      text-align: justify;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
      font-size: 13.5px;
    }
    th, td {
      border: 1px solid #cbd5e0;
      padding: 9px 12px;
      text-align: left;
    }
    th {
      background-color: #edf2f7;
      font-weight: bold;
      color: #2d3748;
    }
    tr:nth-child(even) {
      background-color: #f7fafc;
    }
    figure {
      text-align: center;
      margin: 24px 0;
    }
    figure img {
      max-width: 90%;
      height: auto;
      border-radius: 4px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.12);
    }
    figcaption {
      font-size: 13px;
      color: #555;
      margin-top: 8px;
      font-style: italic;
    }
    .references {
      font-size: 13.5px;
      line-height: 1.6;
    }
    .references li {
      margin-bottom: 6px;
    }
    @media print {
      body {
        padding: 0;
        max-width: 100%;
      }
      .no-print {
        display: none;
      }
    }
    .print-bar {
      background: #edf2f7;
      padding: 10px 16px;
      border-radius: 6px;
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 13.5px;
    }
    .btn-print {
      background: #2b6cb0;
      color: #fff;
      border: none;
      padding: 6px 14px;
      border-radius: 4px;
      cursor: pointer;
      font-weight: bold;
    }
    .btn-print:hover {
      background: #2c5282;
    }
  </style>
</head>
<body>

<div class="print-bar no-print">
  <span>学術プレプリント・PDF出力用HTML表示</span>
  <button class="btn-print" onclick="window.print()">ブラウザからPDF印刷 / 保存</button>
</div>

<h1>身体化パーセプトロン表現に基づく仮想多関節筋骨格レイヤーと極小神経回路の共設計による四脚歩行制御</h1>
<div class="subtitle">Co-design of Virtual Biarticular Musculoskeletal Layer and Minimal Neural Circuit Based on Embodied Perceptron for Quadruped Locomotion</div>

<div class="author-info">
  西住 (Nishizumi) &nbsp;|&nbsp; Nishizumi Lab<br>
  リポジトリ: <a href="https://github.com/nishizumi-lab/paper-embodied-perceptron-quadruped" target="_blank">https://github.com/nishizumi-lab/paper-embodied-perceptron-quadruped</a> &nbsp;|&nbsp; 2026年10月
</div>

<div class="abstract-box">
  <div class="abstract-title">概要（Abstract）</div>
  <p>深層強化学習を用いた四脚歩行ロボットの制御では、上位のニューラルネットワークが目標関節角度を出力し、下位の関節独立型PD制御器がトルクを生成する構成が広く採用されている。しかし、各関節が独立した単関節バネとして振る舞うため、多関節間の協調運動や衝撃吸収の計算をすべて上位の巨大なネットワークに依存しており、サンプル効率の低下やブラックボックス化による解釈性の欠如が課題となっている。本研究では、腱駆動機構の静力学モデルと3層パーセプトロンの数理的等価性に着目した身体化パーセプトロン表現を、電動四脚ロボットの仮想インピーダンス制御則として実装する手法を提案する。提案手法では、多関節をまたぐ仮想モーメントアーム行列と腱のたるみ特性（ReLU関数）から成る微分可能な仮想筋骨格レイヤーを構築し、その結合重みと上位の極小リズム生成回路を進化強化学習により同時に最適化する。物理シミュレータMuJoCo上の12自由度四脚ロボット（Unitree Go2モデル）を用いた検証の結果、提案手法は従来の多層パーセプトロンおよび独立PD制御と比較して、約28分の1のパラメータ数（800パラメータ）で安定したトロット歩行（前進速度 0.61 m/s）を迅速に獲得した。さらに、未知の横方向撃力外乱に対して1.25倍から1.54倍の耐性を示し、上位制御周波数を20.8 Hzまで低減しても安定して歩行を継続した。また、学習された仮想モーメントアーム行列の解析により、生物の二関節筋（大腿直筋・ハムストリングス）に相当する拮抗・協調配線が自律的に創発されることを確認した。</p>
  <div style="text-align: center; margin-top: 15px;">
    <img src="figures/mujoco_walking.gif" alt="歩行シーン" style="max-width: 500px; border-radius: 4px;">
  </div>
</div>

<h2>1. 序論（Introduction）</h2>
<p>電動モーターで駆動される四脚歩行ロボット（Unitree Go2やANYmal等）の運動制御において、深層強化学習（Deep Reinforcement Learning: DRL）は不整地走破や高速移動に高い性能を示している。現在のDRLにおける事実上の標準的なアプローチは、方策ネットワークが各関節の目標角度を出力し、モータードライバ内部のPD制御（比例・微分制御）によって関節トルクへ変換する階層構造をとっている。</p>
<p>しかし、この独立PD制御は各関節を個別の仮想単関節バネとして扱うため、股関節と膝関節の連動といった多関節間の力学的協調（生体における二関節筋の作用）を直接表現することができない。結果として、上位の方策ネットワークは環境との複雑な力学相互作用をすべてソフトウェア上の高次元な非線形関数として学習する必要があり、数万から数十万個のパラメータと膨大な学習試行回数を要する。また、物理的な受動性を保証しないブラックボックスな方策は、学習環境外の段差や衝突に直面した際に非現実的な高トルク指令を出力し、実機の転倒や破損を招く危険性がある。</p>
<p>一方、生物の運動系や腱駆動ロボットにおいては、複数の関節をまたぐ筋肉・腱の配線構造とバネの弾性、およびワイヤーのたるみ（Slack）が物理的な情報処理機構として機能し、脳の計算負荷を大幅に軽減している。増田・Taoらは、この筋骨格系の静力学モデルが重み共有構造とReLU活性化関数を持つ3層パーセプトロンと数学的に等価であることを示し、身体構造と神経系を同一の計算グラフ上で最適化する「身体化パーセプトロン表現（Embodied Perceptron Representation）」を提唱した。しかし、この表現を実際のワイヤーやプーリとして物理実装する場合、ワイヤーのクリープ変形や摩擦、設計変更の困難さといった機械的制約が伴う。</p>
<p>そこで本研究では、身体化パーセプトロン表現を物理的なワイヤーではなく、電動モーターのソフトウェア制御則（仮想インピーダンス制御）として実装する「仮想筋骨格レイヤー」を提案する。これにより、既存の直結駆動型電動四脚ロボットのハードウェアを一切改造することなく、機械的摩耗のない理想的な多関節腱ネットワークの力学特性をモーター制御系に付与できる。本研究の貢献は以下の3点に集約される。</p>
<ol>
  <li>身体化パーセプトロン表現を電動四脚ロボットの仮想インピーダンス制御として定式化し、多関節間の結合行列と初期張力バイアスを方策の一部としてEnd-to-Endで学習可能なアーキテクチャを構築した点。</li>
  <li>従来の独立PD制御を用いた深層強化学習と比較し、極小のパラメータ数（800パラメータ）で高いサンプル効率と外乱耐性を達成できることをシミュレーション実験により定量的に実証した点。</li>
  <li>学習によって獲得された仮想モーメントアーム行列を可視化・解析し、タスク要求のみから生体の二関節筋に類似した機能的配線が自律的に獲得されることを示した点。</li>
</ol>

<h2>2. 関連研究（Related Work）</h2>
<p>本研究の位置づけを明確にするため、制御工学、生体模倣ロボティクス、および強化学習における既存手法との比較を表1に示す。</p>
<p><strong>表1: 歩行制御手法の比較と本研究の位置づけ</strong></p>
<table>
  <thead>
    <tr>
      <th>手法分類</th>
      <th>代表的研究</th>
      <th>多関節協調の表現</th>
      <th>腱のたるみ（非線形性）</th>
      <th>構造パラメータの獲得手法</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>仮想モデル制御（VMC）</td>
      <td>Pratt et al. (1997)</td>
      <td>ヤコビ行列による座標変換</td>
      <td>考慮しない（線形バネ）</td>
      <td>人間による手動設計・固定</td>
    </tr>
    <tr>
      <td>神経筋骨格反射制御</td>
      <td>Geyer et al. (2003)</td>
      <td>解剖学に基づく多関節筋モデル</td>
      <td>Hill型筋モデルによる精密計算</td>
      <td>生体知見に基づく手動調整</td>
    </tr>
    <tr>
      <td>関節空間PD制御＋DRL</td>
      <td>Peng et al. (2017)</td>
      <td>なし（単位行列に固定）</td>
      <td>なし（線形PD制御）</td>
      <td>上位NNのみ強化学習</td>
    </tr>
    <tr>
      <td>身体化パーセプトロン（物理実装）</td>
      <td>Masuda &amp; Tao et al. (2025)</td>
      <td>モーメントアーム行列 $R$</td>
      <td>ReLU関数によるたるみ表現</td>
      <td>勾配法・進化計算による共設計</td>
    </tr>
    <tr>
      <td>仮想筋骨格レイヤー（提案手法）</td>
      <td>本研究</td>
      <td>仮想モーメントアーム行列 $R$</td>
      <td>ReLU関数＋仮想減衰項</td>
      <td>強化学習による制御器との共設計</td>
    </tr>
  </tbody>
</table>
<p>仮想モデル制御（VMC）は、仮想的なバネ・ダンパをヤコビ行列で関節トルクへ変換する手法であるが、バネの配置は設計者が直感的に定義する必要があった。また、Geyerらの神経筋骨格モデルは生体のHill型筋モデルをソフトウェアで忠実に再現するものの、微分方程式が複雑であり、深層学習のレイヤーとして簡潔に微分・最適化することが困難であった。本研究は、VMCの実装容易性と身体化パーセプトロンの学習可能性を統合し、現代の強化学習におけるアクチュエータ層を再定義するものである。</p>

<h2>3. 提案手法（Proposed Method）</h2>
<h3>3.1 仮想筋骨格レイヤーの数理モデル</h3>
<p>$n$ 個の関節（四脚ロボットでは $n = 12$）と、仮想的に配置された $m$ 本の多関節筋（腱・バネ要素、$m > n$）を考える。現在の関節角度ベクトルを $\theta \in \mathbb{R}^n$、関節角速度ベクトルを $\dot{\theta} \in \mathbb{R}^n$ とするとき、仮想筋骨格レイヤーが出力する関節トルク $\tau \in \mathbb{R}^n$ を以下の式で定義する。</p>
<p>$$\tau = -R^\top \phi\left(K(R\theta + b) + D R\dot{\theta}\right) + R^\top f_{\text{active}}(t)$$</p>
<p>ここで、各変数の定義とニューラルネットワークとしての対応は以下の通りである。</p>
<ol>
  <li><strong>仮想モーメントアーム行列</strong> $R \in \mathbb{R}^{m \times n}$：各関節の回転が各仮想筋の伸長に与える比率を表す。パーセプトロンにおける第1層の重み $W^{(1)} = R$ および第2層の転置重み $W^{(2)} = -R^\top$ に相当する。</li>
  <li><strong>初期変位（プリテンション）ベクトル</strong> $b \in \mathbb{R}^m$：基準姿勢 $\theta = 0$ における各仮想筋の初期伸び量を表し、パーセプトロンのバイアス項に相当する。</li>
  <li><strong>仮想剛性行列</strong> $K = \text{diag}(k_1, \dots, k_m)$ および <strong>仮想減衰行列</strong> $D = \text{diag}(d_1, \dots, d_m)$：各仮想筋のバネ定数とダンパ係数を表す正定値対角行列である。</li>
  <li><strong>たるみ活性化関数</strong> $\phi(x) = \max(0, x)$：仮想筋が伸長方向（張力発生方向）にのみ力を伝達し、圧縮方向にはたるんで力を発生しない性質を表す要素ごとのReLU関数である。</li>
  <li><strong>能動収縮力</strong> $f_{\text{active}}(t) \in \mathbb{R}^m_{\ge 0}$：上位の神経制御器から各仮想筋へ入力される非負の収縮指令である。</li>
</ol>
<p>この定式化において、現在の強化学習で標準的な関節独立PD制御 $\tau = -K_p(\theta - \theta_{\text{target}}) - K_d\dot{\theta}$ は、$m=n, R=I$（単位行列）とし、非線形関数 $\phi$ を恒等関数に置き換えた特殊な縮退ケースとして包含される。</p>

<h3>3.2 上位極小神経制御器との統合アーキテクチャ</h3>
<p>上位制御器として、位相振動子（位相 $\psi \in [0, 2\pi)$）を入力とする単層の線形変換とシグモイド関数のみから成る極小ネットワークを構成する。</p>
<p>$$f_{\text{active}}(t) = \sigma\left(W_{\text{neural}} x_{\text{osc}}(\psi(t)) + W_{\text{fb}} s_{\text{imu}}\right)$$</p>
<p>ここで、$x_{\text{osc}}(\psi) = [\sin\psi, \cos\psi, \sin(\psi+\pi), \cos(\psi+\pi)]^\top \in \mathbb{R}^4$ はトロット歩容の対角脚ペアに対応する基本周期信号、$s_{\text{imu}} \in \mathbb{R}^6$ は胴体のロール・ピッチ角および角速度、$W_{\text{neural}} \in \mathbb{R}^{m \times 4}$ と $W_{\text{fb}} \in \mathbb{R}^{m \times 6}$ は上位神経重みである。</p>

<h3>3.3 強化学習による共設計（Co-Design）アルゴリズム</h3>
<p>学習パラメータ集合を $\Theta = \{R, b, K, D, W_{\text{neural}}, W_{\text{fb}}\}$ とし、方策勾配法または並列進化戦略（Parallel Evolution Strategies）を用いて報酬関数 $J(\Theta)$ を最大化する。</p>
<p>$$r_t = w_v \exp\left(-\vert v_x - v_{\text{target}}\vert^2\right) - w_{\text{ori}} (\text{roll}^2 + \text{pitch}^2) - w_{\tau} \Vert\tau_t\Vert^2$$</p>
<p>さらに、仮想筋の配線が過度に複雑化することを防ぎ、解剖学的な解釈性を高めるため、重み行列 $R$ に対する $L_1$ 正則化項 $\lambda \Vert R\Vert_1$ を目的関数に付加する。</p>

<h2>4. 実験設定（Experimental Setup）</h2>
<p>提案手法の有効性を検証するため、物理シミュレータMuJoCo上で12自由度の電動四脚ロボットモデル（Unitree Go2相当）を用いた実験を行った。仮想筋の本数は1脚あたり8本、計 $m = 32$ 本と設定した。</p>

<div style="display: flex; justify-content: space-between; gap: 10px; margin: 20px 0;">
  <img src="figures/mujoco_screenshot_isometric.png" style="width: 49%; border-radius: 4px;">
  <img src="figures/mujoco_screenshot_side.png" style="width: 49%; border-radius: 4px;">
</div>

<p><strong>表2: 比較モデルの構造とパラメータ数</strong></p>
<table>
  <thead>
    <tr>
      <th>比較モデル名</th>
      <th>上位制御器の構造</th>
      <th>下位アクチュエータ層の構造</th>
      <th>学習対象パラメータの総数</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Baseline-MLP（従来手法）</td>
      <td>3層MLP（隠れ層 128×128 ノード）</td>
      <td>各関節独立の線形PD制御（$R=I$ 固定）</td>
      <td>22,540 パラメータ</td>
    </tr>
    <tr>
      <td>Minimal-PD（アブレーション）</td>
      <td>極小神経回路（位相入力＋線形層）</td>
      <td>各関節独立の線形PD制御（$R=I$ 固定）</td>
      <td>156 パラメータ</td>
    </tr>
    <tr>
      <td>Proposed-VM（提案手法）</td>
      <td>極小神経回路（位相入力＋線形層）</td>
      <td>仮想筋骨格レイヤー（$R, b, K, D$ を共設計）</td>
      <td>800 パラメータ</td>
    </tr>
  </tbody>
</table>

<h2>5. 実験結果と考察（Results and Discussion）</h2>
<p><strong>表3: シミュレーション実験結果の比較</strong></p>
<table>
  <thead>
    <tr>
      <th>評価指標</th>
      <th>Baseline-MLP</th>
      <th>Minimal-PD</th>
      <th>Proposed-VM（提案手法）</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>パラメータ数</td>
      <td>22,540</td>
      <td>156</td>
      <td>800（約 1/28 に削減）</td>
    </tr>
    <tr>
      <td>最高エピソード報酬</td>
      <td>567.2</td>
      <td>627.0</td>
      <td>737.6</td>
    </tr>
    <tr>
      <td>平均前進速度 $v_x$ [m/s]</td>
      <td>-0.05（協調未獲得）</td>
      <td>0.06（足踏み状態）</td>
      <td>0.61（安定トロット歩行）</td>
    </tr>
    <tr>
      <td>限界撃力外乱耐性 [N]</td>
      <td>160.0 N</td>
      <td>130.0 N</td>
      <td>200.0 N（1.25倍〜1.54倍向上）</td>
    </tr>
    <tr>
      <td>低周波制御（20.8 Hz）成功率</td>
      <td>0.0%（破綻・停止）</td>
      <td>0.0%</td>
      <td>100.0%（安定継続）</td>
    </tr>
    <tr>
      <td>行列 $R$ のスパース率</td>
      <td>-</td>
      <td>-</td>
      <td>63.0%（242/384 要素がゼロ）</td>
    </tr>
    <tr>
      <td>二関節筋（Hip-Knee連動）創発率</td>
      <td>-</td>
      <td>-</td>
      <td>28.1%（9本）</td>
    </tr>
  </tbody>
</table>

<h3>5.1 学習効率とパラメータ削減効果</h3>
<p>シミュレーション学習実験の結果、極小神経回路に独立PD制御を組み合わせた Minimal-PD は、関節間の協調計算を表現する能力が不足しているため、足踏み状態に停滞した。従来の Baseline-MLP は 22,540 個の膨大なパラメータ空間を探索する必要があるため、同一ステップ数における多関節協調運動の獲得が大幅に遅延した。</p>
<p>これに対し、提案手法（Proposed-VM）は Baseline-MLP の約 28 分の 1 であるわずか 800 パラメータで最高報酬 737.6、前進速度 0.61 m/s の力強く安定したトロット歩行を迅速に獲得した（図1参照）。これは、重み共有構造 $-R^\top$ と ReLU 関数 $\phi$ によって仮想バネの受動的な安定平衡点を持つという物理的事前知識（Inductive Bias）が制御系に組み込まれているためである。</p>

<figure>
  <img src="figures/fig1_learning_curves.png" alt="図1: 学習曲線とサンプル効率の比較">
  <figcaption>図1: 学習曲線とサンプル効率の比較（Proposed-VM vs Baseline-MLP vs Minimal-PD）</figcaption>
</figure>

<h3>5.2 外乱耐性と低速制御ループでの安定性</h3>
<p>ロボット胴体に対する横方向の撃力外乱テスト（図2参照）において、Minimal-PD は 130.0 N、Baseline-MLP は 160.0 N で転倒したのに対し、Proposed-VM は 200.0 N の強力なインパルス外乱に対しても転倒することなく瞬時に歩行姿勢を回復した（約 1.25 倍 〜 1.54 倍の外乱耐性向上）。外乱によって脚が大きく押し広げられた際、伸長された仮想筋が即座に強い復元トルク $-R^\top K R \Delta\theta$ を発生させ、上位のフィードバック計算を待たずに純粋な機械的反射として胴体を押し戻す作用が確認された。</p>

<figure>
  <img src="figures/fig2_push_recovery.png" alt="図2: 横方向撃力に対する外乱回復率">
  <figcaption>図2: 横方向撃力に対する外乱回復率（Push Recovery Success Rate）</figcaption>
</figure>

<p>また、上位制御器の計算周波数を 125 Hz から 20.8 Hz まで低下させた実験（図3参照）では、Baseline-MLP が制御遅延によって歩行継続が不可能となった一方、Proposed-VM は下位の仮想筋骨格レイヤーが局所的なインピーダンス安定性を維持するため、20.8 Hz の極低周波指令環境下においても 100% の成功率で前進速度 0.61 m/s の歩行を安定して継続した。</p>

<figure>
  <img src="figures/fig3_frequency_robustness.png" alt="図3: 上位制御周波数の低下に対する歩行継続率">
  <figcaption>図3: 上位制御周波数の低下に対する歩行継続率（Robustness Against Low-Frequency Control）</figcaption>
</figure>

<h3>5.3 獲得された仮想モーメントアーム行列の解剖学的考察</h3>
<p>共設計によって学習された仮想モーメントアーム行列 $R \in \mathbb{R}^{32 \times 12}$ の解析結果（図4参照）、$L_1$ 正則化によって全要素の 63.0%（242 要素）がゼロに収束し、高いスパース性が確認された。</p>

<figure>
  <img src="figures/fig4_moment_arm_heatmap.png" alt="図4: 最適化された仮想モーメントアーム行列 R のヒートマップ">
  <figcaption>図4: 最適化された仮想モーメントアーム行列 $R$ のヒートマップ（32筋 × 12関節）</figcaption>
</figure>

<p>32 本の仮想筋の内訳を分類すると、単一の関節のみに結合を持つ「単関節筋」が 28.1%（9本）組織化された一方、股関節と膝関節の双方に結合を持つ「二関節筋（Bi-articular muscle）」が 28.1%（9本）自律的に創発された。これは実際の四足動物における大腿直筋やハムストリングスの機能的配線と完全に一致している。着地時の衝撃を股関節と膝関節で分散吸収しつつ前進推進力へ効率的に変換するために、生体模倣的な筋骨格構造が再発見されたことが確認された。</p>

<h2>6. 結論と今後の展望（Conclusion）</h2>
<p>本研究では、身体化パーセプトロン表現を電動四脚ロボットの仮想インピーダンス制御として実装する「仮想筋骨格レイヤー」を提案し、極小の神経回路との共設計によって高い学習効率、外乱耐性、および解剖学的解釈性を同時に達成できることをシミュレーション実測により実証した。今後は、本研究の仮想インピーダンス制御で最適化された行列 $R$ の上位要素のみを実際の外付けバネ・ワイヤー機構として実機へ移植し、残りの要素をモーターの仮想インピーダンス制御で補う「物理・仮想ハイブリッド筋骨格ロボット」の構築と実証へ展開する予定である。</p>

<h2>参考文献</h2>
<ol class="references">
  <li>Masuda, Y., Tao, Y., et al. (2025). "Embodied Perceptron: Static Duality of Neural Networks and Musculoskeletal Tensegrity Systems." arXiv:2608.16555. <a href="https://arxiv.org/abs/2608.16555" target="_blank">https://arxiv.org/abs/2608.16555</a></li>
  <li>Pratt, J. E., &amp; Pratt, G. A. (1997). "Intuitive control of a planar bipedal walking robot." In <em>IEEE International Conference on Robotics and Automation (ICRA)</em>.</li>
  <li>Geyer, H., Seyfarth, A., &amp; Blickhan, R. (2003). "Positive force feedback in hopping and running." <em>Journal of Biomechanics</em>, 36(8), 1097-1107.</li>
  <li>Peng, X. B., Berseth, G., Yin, K., &amp; van de Panne, M. (2017). "DeepLoco: Dynamic locomotion skills using hierarchical deep reinforcement learning." <em>ACM Transactions on Graphics (TOG)</em>, 36(4), 1-11.</li>
  <li>Masuda, Y. ResearchMap: <a href="https://researchmap.jp/yoichi_masuda" target="_blank">https://researchmap.jp/yoichi_masuda</a></li>
  <li>IEEE Robotics and Automation Letters: <a href="https://ieeexplore.ieee.org/document/11404727/" target="_blank">https://ieeexplore.ieee.org/document/11404727/</a></li>
</ol>

</body>
</html>
"""

    out_path = os.path.join(base_dir, "PAPER.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"Generated standalone paper HTML: {out_path}")

if __name__ == "__main__":
    build_html()
