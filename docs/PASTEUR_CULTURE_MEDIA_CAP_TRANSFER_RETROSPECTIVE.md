# Pasteur 培養液ボトル・キャップ除去／保持搬送の振り返り

作成: 2026-09-04
対象の実機試行: 2026-08-06（以下、キャプチャ directory 名の時刻は UTC、会話・live-code の時刻は JST）

## 先に結論

覚えている操作は、培養液ボトル上の白いキャップを右手で**横から挟み、ボトル口から上へ外し、保持したままホーム側へ搬送する**ものだった。

これは、次の範囲では実機証拠を伴って成立している。

| 主張 | 状態 | 根拠 |
| --- | --- | --- |
| 白いキャップを選び、開いた二つの爪の間へ位置合わせした | 実証済み | 初回ユーザー tap、キャップと色付きボトル首部の関係、固定 head view の jaw 中心 |
| 爪を閉じてキャップを横から把持した | 実証済み | jaw segment 内判定、開度 0.2006、色付きボトル本体の移動が小さいこと |
| キャップをボトルから外した | 実証済み | 右手先端が 9.175 mm 上がり、ボトル口が head 画像に露出し、元の位置の白い成分が減少 |
| 把持を維持して搬送した | 実証済み | lift 後から 107.415 mm 移動しても開度が 0.2014 から 0.2011 に保たれた |
| 指定位置へ安全に置いた | **未実証** | release 自体は試したが、支持面上のキャップと垂直退避後の非追従を画像／深度で確認していない |
| ねじを回して緩めた（threaded-cap の開栓） | **未実証** | 当時・現行とも、回転／torque／thread を指令・計測する実装がない |
| 再装着して閉栓した | **未実証** | 再装着、回転、密閉性のいずれも試行・検証されていない |

したがって論文では、この実演を **culture-media cap removal and held transfer**（培養液ボトル・キャップの除去と保持搬送）と呼ぶ。キャップを物理的に外したため、機能的にはボトル口へのアクセスを実現したと言える。しかし、`opening and closing a screw cap`、`resealing`、`safe placement` と表現してはならない。

## この記録の根拠と監査範囲

後から Petri dish、SAM、ACT、scene reconstruction 関連の変更が大きく入っているため、現在のコードだけで過去の成功を説明しない。以下を相互に照合した。

| 種別 | 根拠 |
| --- | --- |
| 実機状態 | 2026-08-06 の immutable head RGB-D capture 3 枚、各 `manifest.json` に保存された右手先端姿勢・gripper 開度 |
| 画像上の事実 | 把持前、10 mm lift 後、保持搬送後、release 後の head RGB 画像 |
| 当時の実装 | `fc831c0`（固定 head の cap grasp）と `1f07761`（cap transfer の昇格）の Git tree |
| 当時の試行過程 | アクセス制御された 2026-08-06 Codex event log、そこから実行された runner、RGB-D capture、patch と commit の時刻照合 |

event log では 16:35–18:37 JST の範囲を調べた。cap 関連として成功した patch 適用は 38 回で、主に `media_cap_target.py`、`rgbd_target_scene.py`、`run_culture_media_cap_grasp.py`、task profile、テスト、最終 audit に入っている。これは「最初から完成した一つの script を実行した」記録ではなく、視覚的な誤認識・recess の幾何・把持後の検証を実機で切り分け、通った部分だけを最終フローに残した記録である。認証情報に該当する event-log 行は照合・表示・本書から除外した。

2026-09-04 時点で、`1f07761` から `HEAD` までの cap 関連 source、profile、test には差分がないことを確認した。従って、以下でいう「現在の再現可能な実装」は、後から別タスクのロジックを混ぜたものではなく、当時に昇格した cap-transfer 実装を指す。

## 実機で何が観測されたか

三つの capture は、同じ固定 head view と robot state を使った、昇格対象の証拠列である。`manifest.json` の `robot_state.before` は各画像を取得した瞬間の実測値であり、capture 自体が新しい motion command を送ったという意味ではない。motion の履歴は当日の live runner／event log、状態の確認はこの immutable capture で分担している。

| 段階 | 右手先端の実測 | gripper 開度 | 観測・判定 |
| --- | --- | --- | --- |
| 把持保持、lift 前 | `(0.26880, 0.09134, 0.73435) m` | 0.20057 | tap は二爪を結ぶ線分内。線分からの垂直距離は jaw span の 0.1574 倍 |
| 10 mm lift 後 | `(0.26904, 0.09366, 0.74352) m` | 0.20143 | 実測上昇量 9.175 mm。ボトル本体の正規化移動 0.0095、元位置の白い成分比 0.3757 |
| 保持搬送後 | `(0.30663, 0.04087, 0.82919) m` | 0.20114 | lift 後からの距離 107.415 mm。開度 drift 0.00029、元位置の白い成分比 0.3787 |

ここでいう「元位置の白い成分比」は、tap を中心とする半径 0.30 jaw-span の局所領域に残った低彩度・高輝度成分の比である。これは cap が元のボトル口に残っていないことの補助証拠であり、単独の物体認識ラベルではない。把持・脱離の判断には、これに加えて jaw 間幾何、gripper aperture、色付きボトル本体の静止、手先の実測移動を同時に要求した。

画像は次の順に確認できる。

- [lift 前: キャップを二爪の間で保持](../data/captures/pasteur/2026-08-06/20260806T090454.207716Z_head_culture_media_cap_hold_before_lift_ce98d39c/derived/head_rgb_landscape.png)
- [lift 後: ボトル口の露出とキャップの上昇](../data/captures/pasteur/2026-08-06/20260806T090516.450533Z_head_culture_media_cap_lift_probe10_792a8b3c/derived/head_rgb_landscape.png)
- [保持搬送後: 開度を維持したまま source から離れた状態](../data/captures/pasteur/2026-08-06/20260806T090638.703002Z_head_culture_media_cap_transport_home_hold_c01dae62/derived/head_rgb_landscape.png)
- [release 後: gripper は開いたが、キャップの支持面上の安定を立証できない](../data/captures/pasteur/2026-08-06/20260806T091320.073893Z_head_culture_media_cap_placed_released_4b5635ce/derived/head_rgb_landscape.png)
- [退避後: 同じ理由で placement 成功とは扱わない](../data/captures/pasteur/2026-08-06/20260806T091401.642185Z_head_culture_media_cap_placed_retracted_validation_7077bf98/derived/head_rgb_landscape.png)

offline audit を再実行すると、`accepted: true`、`promotion_scope: cap_side_pinch_lift_and_held_transport`、`placement_promoted: false` を返す。閾値は lift の最小 8 mm、搬送の最小 50 mm、把持開度 0.10–0.40、開度 drift 最大 0.05、ボトル本体の正規化移動最大 0.10 である。

## 何を作り、なぜ通るようになったか

### 1. 最大の白領域ではなく、ボトルに属するキャップを選ぶ

白い物体だけを選ぶと、gripper の白い mount、tag、機材や phone の明るい部分を拾う。`MediaCapTargetAdapter` は次の順に候補を絞る。

1. 色付き（pink/red 系）のボトル首部を見つける。
2. その真上にある、小さくほぼ円筒的な白い成分を cap 候補にする。
3. 初回にユーザーが tap した正規化座標を immutable identity anchor として持つ。
4. jaws が首部を隠した後は、anchor の近傍だけで白い成分を追跡する。遠方の白い gripper mount が「首部の上」に見える誤認識を避ける。

つまり、色だけでも、最大面積でも、固定 pixel ROI でもない。「特定のボトルに支持された cap」という関係と、最初の user identity を組み合わせる。source ROI や許容変位も画素数ではなく、その時点の jaw span／画像対角長で正規化されている。

### 2. 実際の recess を scene と経路に反映する

当初の scene はボトルを台の上に単純に置いたように扱っており、実機のように**同じ高さの二つの platform の間に recessed している**関係を表していなかった。会話で訂正された後、fresh head RGB-D から cap top と bottle body を更新し、ボトルは observed cap top から下へ completion するものとして記録した。

そのため、cap 高さでは横移動せず、まず自由空間で横方向を合わせてから垂直に降りる。接触後は、実機で通った open-jaw approach を逆向きに使う。ただし contact-height waypoint は戻らない。先に 10 mm の vertical lift を確認してから、clearance を保つ reverse egress へ進むためである。

### 3. 「閉じた」ではなく「キャップを挟んだ」を判定する

close は一回だけ送る。次の三つが同時に通らなければ lift を送らない。

1. immutable tap が二つの jaw center を結ぶ線分の中にあり、線分からの距離が jaw span の 0.25 倍以下。
2. gripper aperture が calibrated non-empty band の 0.10–0.40 に残る。
3. cap の下にある色付きボトル本体の移動が、その対角長で正規化して 0.10 以下。

これにより、「空のまま全閉した」「ボトルごと押した」「白い別物を挟んだ」を同じ grasp 成功として扱わない。

### 4. lift、保持搬送、placement を別の状態にする

最終 state machine は次である。

```text
observe → approach-open → align-fixed-head → close
       → lift-probe → verify-removal → transport-hold

select-support → plan-clearance → descend-held → open
       → verify-on-support → retract       # これは未昇格
```

lift では、手先の実測上昇、head view の jaw 移動、source の白い成分減少、ボトル本体の静止、non-empty aperture を同時に検査する。transport では、十分な手先移動と aperture の持続を再検査する。MuJoCo が既に持ち上げた状態の exact route を誤って collision と見なす場合だけは、waypoint hash が hardware evidence と完全一致する場合に限り、その false positive を上書きできる。新規経路や少し変えた経路を許す例外ではない。

placement は、実際に open command を送っただけでは昇格しない。新しい支持面の選択、cap bottom の clearance、open 後の cap-on-support、垂直 retract で cap が追従しないことが必要であり、これらが得られていないので current runner は transport 成功から placement を自動追加しない。

## 当日の会話・live code 監査

時刻は JST。ここでは「今の source がもっともらしい」ことではなく、当日どの仮説を実機で試して捨て、どれを commit に残したかを示す。

| 時刻 | 当時の試行・保存物 | 得られた判断 | 最終フローへの帰結 |
| --- | --- | --- | --- |
| 16:35–16:44 | `media_cap_target.py` / adapter / task profile を導入し、固定 head observation と初期 tap を作成 | 白い候補だけでは gripper mount や遠い明部と混同する | 白色のみ・最大領域・固定 ROI を採用せず、cap-above-coloured-neck + tap identity にする |
| 16:44–17:06 | 複数の Record3D capture、scene refresh、coarse probe | 実機のボトルは二つの同高さ platform の間に recessed しており、汎用 thin-lid の高い lift／単純 scene は合わない | `recessed_between_two_equal-height_platforms` と、cap 専用の fixed-head approach を task profile に固定 |
| 17:20–17:37 | open jaws の lateral alignment、段階的な vertical descent、occlusion 後の backtrack、closed pre-lift capture | jaw が近づくと首部が隠れる。目標そのものを追うだけでは不安定だが、下の色付き bottle body は support anchor になる | immutable tap、local white track、support-motion guard、jaw midpoint servo に分離。grasp の結果は開度約 0.201 で確認 |
| 18:04–18:06 | `hold_before_lift` → `lift_probe10` → `transport_home_hold` の capture 列 | 9.175 mm lift、107.415 mm held transport、開度ほぼ不変、bottle body の過大な追従なし | この三枚と状態値を、removal/held-transfer の immutable evidence として昇格 |
| 18:09–18:14 | descent、clearance、pre-open、release、retract の capture | gripper を開くこと自体は行えたが、cap が指定支持面に残ったことを確認できない | release を placement success と数えない。placement workflow は unpromoted に分離 |
| 18:29–18:37 | audit を 3 回、profile/runner/test/contract を更新して `1f07761` を commit/push | hardware evidence が通る exact egress route のみ reusable とし、未検証 placement は除外 | 現在の `cap_side_pinch_lift_and_held_transport` のみに promotion scope を限定 |

この間、cap runner 呼び出しは 56 回、RGB-D capture は 31 回、cap 関連 test は 8 回、offline audit は 3 回確認できる。回数自体を性能指標にはしないが、失敗・観察・修正・再検証を区別せず一回の成功物語にしていないことの裏付けになる。

## 現在の再現可能な source とテスト

中心となる source は次の通りである。

- [`rollout/media_cap_target.py`](../rollout/media_cap_target.py): cap と bottle 支持関係、tap による identity、jaw geometry。
- [`src/run_culture_media_cap_grasp.py`](../src/run_culture_media_cap_grasp.py): home gate、fresh RGB-D scene、固定 head alignment、close、lift、hold transport の executor。
- [`rollout/cylindrical_cap_transfer.py`](../rollout/cylindrical_cap_transfer.py): robot I/O を持たない evidence gate。
- [`src/configs/pasteur_culture_media_cap_grasp.json`](../src/configs/pasteur_culture_media_cap_grasp.json): normalized threshold と、placement を未昇格とする task contract。
- [`src/audit_cylindrical_cap_transfer.py`](../src/audit_cylindrical_cap_transfer.py): immutable capture の再監査 CLI。

2026-09-04 に次を再実行した。

```bash
/home/admin/miniforge3/envs/robot-test/bin/python src/audit_cylindrical_cap_transfer.py \
  --task-profile src/configs/pasteur_culture_media_cap_grasp.json \
  --before data/captures/pasteur/2026-08-06/20260806T090454.207716Z_head_culture_media_cap_hold_before_lift_ce98d39c \
  --lift data/captures/pasteur/2026-08-06/20260806T090516.450533Z_head_culture_media_cap_lift_probe10_792a8b3c \
  --transported data/captures/pasteur/2026-08-06/20260806T090638.703002Z_head_culture_media_cap_transport_home_hold_c01dae62

/home/admin/miniforge3/envs/robot-test/bin/python -m pytest -q \
  tests/test_media_cap_target.py tests/test_cylindrical_cap_transfer.py
```

結果は audit accepted、`promotion_scope=cap_side_pinch_lift_and_held_transport`、`placement_promoted=false`、test は **9 passed, 2 skipped** だった。skip は現在 `/tmp/pasteur_home_views/head_home.png` がないための hardware-observation test であり、immutable transfer capture を使う integration test は通過している。

cap 関連 source には、現在も `twist`、`torque`、`thread`、`screw`、`unscrew`、`reseal` を実行・検査する経路がない。`culture_cap_close_at_fixed_tap` はボトルを閉める操作ではなく、gripper を閉じて横から把持する stage 名である。この区別は、後の機能追加や論文の表現でも保つ。

## 開扉タスクと一本の論文にするなら

二つを「二種類の完全な laboratory routine」として売るのはまだ早い。より正確で強い中心主張は、**異なる接触構造に対し、視覚と robot state から task-scoped evidence gate を作り、実証された部分だけを再利用可能な policy として昇格する**ことである。

| 観点 | インキュベータ開扉 | ボトル・キャップ除去／保持搬送 |
| --- | --- | --- |
| 接触対象 | articulated appliance の recessed handle | upright removable cap と固定 bottle body |
| 主要動作 | contact-relative pull | horizontal side pinch → vertical detachment → free-space transfer |
| identity の担い手 | registered RGB-D door plane、デモ contact frame、限定的な補助視覚補正 | cap-above-coloured-neck 関係と immutable user tap |
| 接触直後の証拠 | non-empty aperture と 5 mm proof | cap が jaws 間、non-empty aperture、bottle body が静止 |
| 終端の証拠 | registered RGB-D open/closed endpoint | source clearance、9.175 mm lift、107.415 mm held transport |
| 実証外 | 長い pull 中の完全な保持、反復成功率 | placement、再栓、thread torque、反復成功率 |

論文の安全な仮題と一文の主張は、次のようになる。

> **Evidence-Gated Contact Manipulation for a Small Laboratory Cell** — Rather than treating a plausible close command or a segmented object as success, the system promotes only manipulation prefixes supported by task-specific visual and robot-state evidence. We demonstrate an articulated-door endpoint transition and a removable-cap detachment/held-transfer transition.

実験として追加すべきものは明確である。

1. 各 task の独立反復、初期姿勢・照明・物体位置を変えた成功率と失敗分類。
2. identity ablation: white-only / 最大領域 / tap なし / semantic relation + tap の比較。
3. contact ablation: aperture のみ、jaw geometry のみ、support-stationary を加えたものの比較。
4. cap の placement を fresh support selection → cap-on-support → vertical retract まで実証する。
5. screw cap を主張するなら、回転角、トルクまたは slip、口の露出、再装着後の閉栓を別の protocol として追加する。

統合論文の章立て、主張の上限、図と表の案は [PASTEUR_DOOR_CAP_PAPER_OUTLINE.md](PASTEUR_DOOR_CAP_PAPER_OUTLINE.md) に切り出した。実験が増えても、ここで定めた「実証済みの prefix と未実証の suffix を混同しない」という原則を維持する。
