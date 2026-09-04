# Pasteur 開扉・キャップ操作を統合する論文アウトライン

作成: 2026-09-04
根拠となる監査記録:
[インキュベータ開扉](PASTEUR_INCUBATOR_DOOR_OPENING_RETROSPECTIVE.md) と
[培養液ボトル・キャップ除去](PASTEUR_CULTURE_MEDIA_CAP_TRANSFER_RETROSPECTIVE.md)

## 論文の核

この論文の核は「大規模な vision model が lab task を端から端まで解いた」ではない。**contact-rich manipulation では、もっともらしい視覚検出や close command を成功として扱わず、task ごとに必要な観測を evidence gate にして、検証済みの動作 prefix だけを reusable policy に昇格する**、という方法である。

二つの case study は同じ失敗を別の形で示す。

- インキュベータ扉: `gripper が少し開いたまま` や `黒い影が handle に見える` だけでは、扉が開いたとも、最後まで把持できたとも言えない。
- ボトルキャップ: `白いものを閉じた爪が挟んだ` だけでは、キャップを掴んだとも、ボトルを押していないとも、置けたとも言えない。

そこで、semantic identity、接触の物理的 proxy、task endpoint を分け、各 transition に必要な証拠を明文化する。

## 主張の強さを固定する

| 表現 | 採用可否 | 理由 |
| --- | --- | --- |
| 「RGB-D と robot state を用いた evidence-gated contact manipulation」 | 可 | 両 task で実装・監査済み |
| 「インキュベータの開扉 endpoint を確認した」 | 可 | registered RGB-D endpoint が `open` を返す実機 run がある |
| 「キャップをボトルから外して保持搬送した」 | 可 | jaw geometry、9.175 mm lift、107.415 mm transport、aperture persistence がある |
| 「扉を最後まで確実に把持した」 | 不可 | 成功 run では full pull 中に aperture が空把持へ移った |
| 「キャップを指定位置に置いた」 | 不可 | post-release support evidence がない |
| 「ボトルを開栓・閉栓した」 | 不可 | rotation / torque / reseal の実装・証拠がない |
| 「一般の lab にそのまま転移する」 | 不可 | まだ一つの cell と少数の hardware evidence に限られる |

この制約を守るほうが、後で placement／screw manipulation を足した際に主張が自然に強くなる。

## 推奨タイトル

1. **Evidence-Gated Contact Manipulation in a Small Laboratory Cell**
2. **From Plausible Perception to Verified Manipulation Prefixes for Laboratory Robots**
3. **Task-Scoped Visual and Robot-State Evidence for Contact-Rich Laboratory Manipulation**

1 が最も短く、現状の実証範囲と合う。

## 論文構成案

### 1. Introduction

- Lab automation では認識が合って見えても、接触、滑り、遮蔽、支持物の移動で実行が破綻する。
- 成功条件を一つの detector score に畳み込まず、操作の状態遷移ごとに、必要な視覚／robot-state evidence を課す。
- 2 task を通じて、目標物の形態が異なっても同じ design principle が働くことを示す。

### 2. Method: evidence-gated manipulation prefixes

共通状態機械:

```text
semantic enrollment
  → free-space alignment
  → contact action
  → local contact proof
  → task-specific progress proof
  → endpoint proof or fail-closed recovery
```

共通要素:

1. **Semantic enrollment**: 固定 pixel を保存せず、object/support relation と必要なら一回の user tap で identity を固定する。
2. **Task-scoped geometry**: cap は jaw span、door は contact-frame-relative SE(3) を単位にする。別 task の pixel goal や generic thin-object height を流用しない。
3. **Contact proof**: aperture、jaw 内幾何、support-body motion、短い proof motion を組み合わせる。
4. **Promotion scope**: 成功に見える command sequence ではなく、immutable captures と measured state が通った prefix にだけ reusable label を与える。
5. **Fail-closed suffixes**: endpoint / placement evidence がなければ、次の contact action を自動追加しない。

### 3. Case study A: articulated appliance door

- 成功テレオペを contact frame 相対の pull trajectory にコンパイル。
- live RGB-D plane yaw、bounded visual correction、close → 5 mm proof → checkpointed pull を組み合わせる。
- endpoint は registered RGB-D の open / closed reference で判定する。
- 成功例でも long pull 中の slip があったことを明示し、endpoint success と continuous grasp success を区別する。

### 4. Case study B: removable culture-media cap

- 白色だけではなく `white cap above coloured bottle neck + immutable tap` で identity を定義する。
- bottle の recess geometry を更新し、lateral align → vertical descent → side pinch にする。
- lift 後に source clearance、bottle の静止、aperture persistence を確認し、exact hardware-observed egress のみを再利用する。
- 9.175 mm removal と 107.415 mm held transport は結果として示す。release/placement と screw closure は未実証として分離する。

### 5. Evaluation

現時点の実機結果は case-study evidence であり、成功率の比較実験ではない。投稿前には次を追加する。

| 評価 | 最低限の設計 | 指標 |
| --- | --- | --- |
| 反復性 | task ごとに独立 trial を複数回、初期 pose と照明を変える | stage 別 pass rate、終了状態、recovery rate |
| identity | white-only、最大 component、tap なし、relation + tap | target identity precision、false-contact rate |
| contact gate | aperture のみ、視覚のみ、全 gate | true grasp/removal precision、false-positive rate |
| 経路 | direct motion、evidence-gated prefix、exact-route reuse | contact collision、object push-out、completion time |
| cap placement | fresh support → place → retract | cap-on-support、retract non-follow、placement success |

door の open/close endpoint と、cap の detach/transfer の state transitions を同じ表に並べると、「行動の名前」ではなく「各 transition が何で証明されたか」が比較できる。

### 6. Discussion and limitations

- vision model は semantic enrollment の候補を出せても、contact の成功証明を置き換えない。
- SAM は必要な object identity が曖昧なときだけ使い、cap の最終 gate は SAM score ではなく geometry、support、robot state で持つ。
- MuJoCo は free-space route と collision audit に役立つが、hardware evidence を無条件に上書きしない。false-positive override は exact measured route のみに拘束する。
- 現段階では一つの lab cell の証拠であり、generalization は主張ではなく今後の実験課題である。

## 図・表の最小セット

1. **Figure 1 — System and common state machine**: RGB-D/head・wrist view、robot state、MuJoCo planner、evidence gate の関係。door と cap へ分岐する図。
2. **Figure 2 — Door timeline**: closed frame、contact/proof、open endpoint。途中の aperture slip と fail-closed recovery を明示する。
3. **Figure 3 — Cap timeline**: closed side pinch、9.175 mm lift、107.415 mm held transport。release/placement は破線で「未昇格」とする。
4. **Table 1 — Evidence contract**: 各 state transition、必要な観測、失敗時の行動。
5. **Table 2 — Ablation / repetition results**: 新規データを取ってから埋める。空欄を推定値で埋めない。

## 現在から追加すべき実験データ

### Door

- 初期 door pose、照明、handle visibility を変えた repeated trials。
- full pull 中の slip を減らす contact pose / jaw-force / initial pull comparison。
- open endpoint だけでなく、continuous hold の正解ラベルを別途記録。

### Cap

- source・support の位置を変えた removable-cap trials。
- fresh placement support を選び、cap-on-support と vertical-retract non-follow を確認する置き直し。
- thread を主張したい場合だけ、rotation 量、torsional slip、口の露出、再装着後の closure を新しい task として設計する。side pinch lift の既存データをその証拠に流用しない。

### 共通

- 各 trial について stage-level journal、before/after RGB-D、robot state、失敗理由を保存する。
- 事後に人が目視で成功を選ぶ前に、evidence-gate の閾値と promotion rule を固定する。
- paper の result table では、`endpoint success`、`contact maintained`、`object placed` を一つの success に混ぜない。

## 執筆上の最重要原則

二つのタスクを一つにまとめる価値は、「扉を開けた」「蓋を外した」という二つのデモを並べることだけではない。曖昧な視覚判断から始めても、接触の直前・直後・終端で何を確認すれば次の action を許せるかを、物体形態に応じて具体化した点にある。

このため、実証済みの prefix は強く具体的に書き、未実証の suffix は明示的に止める。その姿勢自体が、後続の placement、screw closure、他 lab への転移を正しく評価できる土台になる。
