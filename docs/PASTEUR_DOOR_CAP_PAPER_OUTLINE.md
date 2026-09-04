# Code as a Learning Machine: How to Use Codex for Robot Control

作成: 2026-09-04

実験根拠:
[インキュベータ開閉の監査](PASTEUR_INCUBATOR_DOOR_OPENING_RETROSPECTIVE.md) と
[培養液ボトル・キャップ除去の監査](PASTEUR_CULTURE_MEDIA_CAP_TRANSFER_RETROSPECTIVE.md)

## 一文でいうと

ニューラルネットの weight だけでなく、**知覚器、座標変換、シミュレータ、制御則、成功判定、テストを含む実行可能なコードそのものを、実機で測った empirical loss によって学習する**。

Codex はこの広い仮説空間を探索する optimizer であり、Git repository は学習される model、robot trial は training sample、画像・深度・関節・gripper 状態は loss を計算する evidence になる。

## Abstract draft

Robot learning usually optimizes the weights of a fixed neural architecture. We study a broader learning substrate: the executable robot repository itself. A code agent observes physical trials, identifies empirical failure, and rewrites perception, calibration, simulation, control, and verification code to reduce an evidence-based loss. Successful hypotheses are tested and committed, turning Git history into a reproducible learning trace. In a small wet-lab cell, Codex produced two contact-rich capabilities: autonomous opening and closing of an incubator door, and removal and held transport of a culture-media bottle cap. The door controller incorporated twelve teleoperated demonstrations into a contact-relative trajectory with visual and mechanical verification. The cap controller used zero cap-specific teleoperation demonstrations and no task-specific neural-network training; it constructed a visual servo and verified transfer program from one target click, seven active pose observations, and empirical trial feedback. The cap was lifted 9.2 mm from the bottle and transported 107 mm while retained by the gripper. These results suggest that code agents can learn at a level above conventional robot policies, selecting and integrating neural perception, geometric vision, simulation, controllers, and tests as task demands change.

## 1. Code に対する empirical risk minimization

通常の robot learning は、固定された network $f_\theta$ の parameter を最適化する。

\[
\theta^* = \arg\min_\theta \hat L_{\mathrm{emp}}(f_\theta; D)
\]

本研究では、学習対象を実行可能な program $P$ に広げる。

\[
P^* = \arg\min_P \hat L_{\mathrm{emp}}(\operatorname{Execute}(P); D)
\]

$P$ は Python source だけではない。config、camera mapping、object representation、MuJoCo model、外部 perception tool、trajectory、failure recovery、unit test、実機 evidence contract を含む。Codex は trial $t$ の結果から patch を提案し、次の program $P_{t+1}$ を作る。

この実験で使った loss は、一つの連続値に無理にまとめるより、次の観測可能な項を持つ constrained empirical loss と考えると分かりやすい。

\[
\hat L = w_i L_{identity} + w_c L_{contact} + w_p L_{progress}
       + w_e L_{endpoint} + w_r L_{runtime}
\]

安全条件と evidence の欠落は hard constraint にする。optimizer が成功判定そのものを書き換えて loss を小さく見せないよう、immutable capture、実測 robot state、独立 endpoint、test を evaluator として保存する。

## 2. 学習ループ

```text
観測する
  → 失敗を測定可能な loss にする
  → Codex が code/config/tool を変更する
  → offline test と simulation を行う
  → bounded robot trial を行う
  → image/depth/robot state で評価する
  → 改善なら commit、違えば次の patch
```

この loop が学習である。Codex は $P$ に対する language-guided、non-gradient の optimizer として働く。最終成果物は chat 中の reasoning ではなく、Codex なしで再実行できる code、config、test、evidence である。

## 3. なぜ weight だけを学ぶより自由なのか

VLA は高次元の視覚入力から滑らかな motor primitive を得るのに強い。一方、未知の lab task で必要になる変更は、しばしば固定 network の入出力より外側にある。code agent は VLA と競合する低レベル policy ではなく、VLA も必要に応じて呼び出せる一段上の学習器である。

| 学習できるもの | 固定 architecture の VLA | Code agent |
| --- | --- | --- |
| sensor と座標系 | 与えられた入力に適応 | camera driver、calibration、RGB-D registration 自体を変更 |
| perception | weight 内の表現 | SAM、古典 CV、depth、tag、VLM を選択・結合 |
| world model | network 内部または固定 simulator | MuJoCo scene を作成・修正し collision test を追加 |
| action | 学習済み action space 内 | controller、state machine、探索、recovery を新しく実装 |
| supervision | 多数の demonstration / reward が中心 | 言語指示、少数 probe、既存 demo、unit test、実機 endpoint を併用 |
| success definition | dataset / reward に埋め込まれる | task ごとの evidence gate として明示・監査できる |
| learned artifact | weight checkpoint | 動く repository、Git diff、test、再現手順 |

重要なのは自由度そのものだけではない。agent が新しい segmentation tool や simulator を導入し、必要なら NN を一部に使い、その出力が本当に task progress につながったかまで code で検査できることである。

## 4. Experiment A — Incubator door opening and closing

この task では、12 本の成功テレオペデモを使った。Codex はデモを丸ごと絶対座標で replay するのではなく、handle contact を原点とする相対 SE(3) trajectory に変換した。さらに、head RGB-D による door-plane yaw、gripper aperture による contact proof、5 mm proof pull、途中 checkpoint、open/closed endpoint classifier を code として追加した。

結果として、ロボットは閉じたインキュベータを自律的に開け、別の dedicated close trajectory で閉められた。これは単なる imitation ではなく、デモ、視覚、機械状態、endpoint を一つの実行可能な program に再構成した例である。

## 5. Experiment B — Demonstration-free cap removal

キャップ task は対照的である。

- cap-specific teleoperation demonstration: **0 本**
- task-specific neural-network training: **0 回**
- user input: 対象を固定する **1 tap** と、途中の自然言語 feedback
- active calibration: **7 個の open-jaw pose observation** から局所 image-to-motion Jacobian を構成

Codex は既存の robot motion primitives を再利用しながら、次を新しく code 化した。

1. `white cap above coloured bottle neck` という object relation と tap identity。
2. 二つの platform の間に recessed した bottle の scene geometry。
3. jaw midpoint を cap に合わせる fixed-head visual servo。
4. bottle body が動いていないこと、cap が jaws 間にあること、non-empty aperture を同時に見る contact loss。
5. 10 mm lift 後の source clearance と、保持搬送後の aperture persistence を見る progress loss。

実機では、cap を 9.175 mm 持ち上げて bottle mouth を露出させ、gripper 開度を約 0.201 に保ったまま 107.415 mm 搬送した。

これは「データを使っていない」という意味ではない。robot trial の画像と状態は empirical data である。しかし、正解 action sequence を示す教師デモも、gradient で更新する task policy もなかった。agent が実験から制御 program と loss evaluator を同時に構成した、**demonstration-free empirical program learning** の例である。

## 6. 二つの実験が示すこと

| | Door | Bottle cap |
| --- | --- | --- |
| 初期 supervision | 12 teleop demonstrations | 0 teleop demonstrations、1 tap |
| Codex が学習したもの | contact-relative trajectory と door-specific verification | semantic adapter、visual Jacobian、grasp/lift/transport verification |
| 実機結果 | autonomous open endpoint と autonomous close endpoint | 9.175 mm removal と 107.415 mm held transport |
| 共通原理 | 実機 evidence を loss として code を反復改善し、成功 program を Git に固定 |

同じ code-learning loop が、デモが豊富な task ではデモを構造化し、デモがない task では active observation から新しい controller を作った。この supervision の柔軟性が、二つを一本の論文にする理由である。

## 7. Contributions

1. **Code-level empirical learning**: robot repository 全体を hypothesis とし、実機 evidence によって program を最適化する見方。
2. **Agentic tool expansion**: perception、depth、simulation、control、test を task に応じて追加・交換できる Codex loop。
3. **Auditable embodied learning**: trial、patch、test、commit を対応付け、何が学習され、なぜ成功したかを再現可能にする方法。
4. **Two complementary demonstrations**: demo-grounded door manipulation と、demonstration-free cap removal を同じ枠組みで実現。

## 8. 最小限の追加実験

簡潔な論文にするなら、実験を増やしすぎず次の三つに絞る。

1. Door と cap を初期位置を変えて複数回実行し、task success と必要 patch 数を測る。
2. `固定 code / VLA-only baseline / Codex code learning` を、成功率、robot trial 数、追加教師デモ数、適応時間で比較する。
3. cap で `colour only → relation → relation + tap → full evidence loss` の ablation を行う。

主結果の表は、最終成功率だけでなく、**新しい task を何回の実機 trial、何本の教師デモ、何回の code revision で獲得したか**を中心にする。これが weight learning との差を最も分かりやすく示す。

## 論文全体の短い構成

1. Introduction: weight だけでなく code を学習対象にする。
2. Method: Codex patch loop と empirical loss。
3. Door: demonstration を program に変換した例。
4. Cap: demonstration なしで program を獲得した例。
5. Evaluation: success、sample efficiency、ablation。
6. Discussion: VLA を code-learning system の一 primitive として統合する。

メッセージは一つでよい。

> **A robot can learn not only by changing numbers inside a fixed policy, but by changing the executable machinery that senses, reasons, acts, and verifies success.**
