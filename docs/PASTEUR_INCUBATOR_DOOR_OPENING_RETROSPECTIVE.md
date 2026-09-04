# Pasteur インキュベータ開扉の振り返り

作成: 2026-09-04  
対象の実機試行: 2026-08-08（以下の時刻は、特記しない限り記録ディレクトリ名の UTC）

## 先に結論

開扉できた理由は、**画像からハンドルを一発で検出して任意の方向へ引いたからではない**。右腕で取得した、視覚的に選別済みのテレオペ開扉デモから「把持した瞬間を原点にした」軌道を作り、ライブのインキュベータ姿勢に合わせてから再生したためである。

実機での自動試行
`data/runs/pasteur/incubator_auto_open_20260808_demo_retry2/` は、閉扉と判定された状態から始まり、最終的に RGB-D の端点判定で `open` を返して `status: success` になっている。実際に、初期画像では前面パネルが閉じ、終端画像では扉の内側が見えている。

ただし、この成功を「最後まで完全に把持を維持した成功」とは呼べない。5 mm の把持テストまでは通過したが、長い引き軌道の途中でアパーチャ監視が空把持を検出して停止した。その時点で扉はすでに開いており、退避後の RGB-D 判定が開扉を確認した、というのが保存済みログと一致する正確な説明である。

## このレポートの根拠と範囲

このリポジトリにはその後の Petri dish、SAM、ACT、テレオペ周りの変更が多数ある。そのため、現在のコードだけから過去の成功理由を推測しないよう、次の三種類を分けて照合した。

| 種別 | 根拠 |
| --- | --- |
| 実機の事実 | `data/runs/pasteur/incubator_*20260808*/` の `journal.json`、アパーチャ記録、RGB-D 証拠画像 |
| デモの事実 | `data/reference/pasteur/incubator/compiled_door_open_v1.json` と参照 HDF5 |
| 実装の事実 | Git 履歴の `2b6ccab` → `676981b` と現行の door 関連モジュール |
| 当時の試行過程 | 2026-08-08 のアクセス制御された Codex event log、そこから起動された stage ごとの run directory、各 patch の Git 帰着先 |

最後の行は、後から書いた説明を当時の意図として扱わないために追加した監査である。会話ログのうち 2026-08-08 12:43–19:08 JST を調べ、成功した patch 適用 58 回（17 ファイル）と、ライブで送った stage コマンドを保存済み run に時刻・名前で照合した。認証情報に該当する行は照合・表示・本書から除外した。この照合で、当時の手探りの試行と、最終的に残した通常ワークフローを区別した。

2026-09-04 時点で、`676981b` から `HEAD` までの door 関連コード・設定・テストには差分がないことを確認した。従って本書の「現在の再現可能な実装」は、この固定された door ワークフローを指す。一方で、成功した自動試行は `283b913` のコミットより約 5 分前なので、その瞬間の未コミット作業ツリーまで完全に復元できるわけではない。実機ログを優先し、Git は設計の復元に用いた。

なお、記録中の MuJoCo ブランチ名 `left_arm_*` は物理左腕を意味しない。物理的な実行対象はプロファイルの `arm: right` と RPC の `set_right_ee_target` であり、カメラ/実機対応は `robot/camera_map.json` を基準にしている。

## 開扉できる前に何が足りなかったか

2026-08-08 14:54 JST より前の Git には、専用の door デモコンパイラ、扉面の RGB-D 推定、把持証明、端点判定、専用実行器が存在しなかった。汎用テレオペ／リプレイだけでは、以下の不確かさを切り分けられなかった。

| 問題 | 初期の実機ログ | 後で採った対策 |
| --- | --- | --- |
| 接触位置・姿勢が一定でない | 12 本の成功テレオペでも接触位置の標準偏差は XYZ = 6.5 / 10.8 / 4.7 mm、全幅は最大 36.1 mm、姿勢誤差の中央値は 4.6° | 絶対座標をそのまま再生せず、接触フレーム相対の SE(3) 軌道にする |
| 閉じてもハンドルを挟めているとは限らない | 初期試行ではアパーチャ 0.684 で不安定、別試行では 0.005（ほぼ全閉＝空把持）で不合格 | 一定時間のアパーチャ列が「部分的に開いたまま安定」して初めて把持候補とする |
| 扉面の向きが微妙にずれる | 手作業で XY/Z/Yaw の探索を何度も繰り返した | 頭部 RGB-D の低彩度な垂直面を RANSAC 平面フィットし、ライブ yaw の残差だけを補正する |
| 画像中の黒い形が信頼できない | ハンドル近傍の黒い影・腕・爪が照明で変わる | 影を把持・接触・開扉の証拠にしない。右カメラの赤い EYELA ラベルは、ハンドルではなく剛体親の微小横補正だけに使う |
| 「扉が開いた」の判定が曖昧 | 初期の `open_in_place` 等には、独立した最終端点判定がない | 固定タグで登録した RGB-D を、保存済みの closed/open 端点と比較する |

アパーチャは `1 = 全開、0 = 全閉` の正規化値である。実際に正しい把持らしいと判定された値はおおむね 0.30–0.41、完全に閉じて空だった失敗は約 0.005 だった。この差を明示的な判定器にしたことが、単に「閉じる」から「ハンドルを保持している可能性が高い」へ進んだ大きな変更である。

## 何を作り、どう開けるようにしたか

### 1. 成功デモを接触基準の軌道へコンパイル

`src/compile_incubator_door_demos.py` は、確認済みの 12 本の右腕テレオペ HDF5 を入力にする。選別の実際の順序は、まず到達時の閉じた gripper 区間の正味 pull で 57 本を順位付けし、次に上位 12 本の**最終 head 画像で扉が開いていることを目視確認**し、その名前をコンパイラ入力として固定した、というものだった。したがってコンパイラ自身が成功ラベルを推論するわけではないが、選別も単なる「大きく引いた軌道」の自動採用ではない。各デモから以下を抽出する。

- jaws が閉じ始めるフレーム
- その 10 フレーム前の preclose
- 接触後 5 mm 動いた proof フレーム
- jaws が再び開く直前の release

接触姿勢の medoid は `door_open_20260703_163756`、`close=56`、`proof=68`、`release=203` だった。接触から proof までは 0.40 秒・正味 5.5 mm、接触から最終 release 前までは 148 サンプル・4.93 秒・正味約 211 mm の軌道である。実行時には contact pose を原点として各 SE(3) オフセットを掛け直すため、インキュベータが少し平行移動・yaw 回転しても、デモの絶対座標には縛られない。

### 2. 接触前の位置合わせ

1. 頭部 Record3D を 8 フレーム取得する。
2. 固定 AprilTag（通常 3, 12, 13）でライブ画像を基準画像へ登録し、扉面の RGB-D 平面を推定する。固定タグ 3 は robot/head の基準、13 はインキュベータ近傍 ROI の種であり、タグそのものを扉面とみなさない。
3. 平面法線の yaw と基準 yaw（-5.38°）の残差だけで、開いた jaws の preclose 姿勢を回す。許容は ±15°。
4. 必要なら右手カメラで最大 3 回だけ、小さな横方向補正を行う。1 回の上限は 15 mm で、赤ラベルが直接「ハンドル位置」を決めることはない。
5. jaws を開いたまま、デモの preclose→contact 相対変換を最後に適用する。これにより微小な視覚補正を接触で打ち消さない。

固定 Pasteur プロファイルのこの経路は、SAM でハンドルをセグメントしていない。SAM+RGB-D は別ラボへの appliance enrollment には使えるが、ここで実証された固定環境の開扉は RGB-D 平面・タグ登録・デモ軌道が中心である。

### 3. 「閉じる」ではなく「証明してから引く」

接触姿勢で一度だけ jaws を閉じ、20 Hz でアパーチャを観測する。以下を満たさなければ full pull は送らない。

1. 末尾 10 サンプルが全閉の空把持上限 0.02 を超え、ばらつきも 0.04 以下である。
2. デモの最初の 5 mm を再生しても、最初の閉じ量の 65% 以上を保持する。
3. さらに静止状態で同じ保持条件を再確認する。

これを通過してから、残りの demo-relative pull を 30 Hz・2 倍の時間スケール（公称約 9.86 秒）で送る。途中 15 フレームごとにアパーチャを確認する。空把持なら、ホーム方向へ動かず、まず 15 mm 退避して jaws を開く。

### 4. 終端は扉そのものから判定

把持の成否だけでは扉が開いたかは分からない。実行後にもう一度頭部 RGB-D を取り、タグ登録後の深度を open / closed の保存済み端点と比較する。閉扉マーカーが見えないことだけを開扉の証拠にはしない。曖昧なら `unknown` として次の接触を送らない。

## 当日の会話・ライブコード監査

これは「最終コードがもっともらしい」ことの確認ではなく、当時 Codex が何を実機で試し、何を失敗として捨て、どの変更が commit になったかの照合である。時刻は JST。各 run directory は当時の CLI 実行と同じ UTC 時刻を含む。

| 時刻 | 実際の試行と保存物 | 確認できた結果 | 現在の通常フローへの帰結 |
| --- | --- | --- | --- |
| 12:34–13:32 | 開爪の `retreat-orient`、`measured-lateral-step`、`demo_hover`、`demo_preclose`。例: `incubator_door_20260808T034702Z_retreat_orient/`、`...T043016Z_demo_hover/` | 開始時の手先を接触 anchor とみなした仮説は誤りだった。会話中の実測比較では成功 contact より約 93 mm 手前だった。また小さな相対指令を連ねると手首角が累積ドリフトした。 | 初期のライブ姿勢を contact 原点にしない。デモ由来 preclose を基準にし、現在の扉面 yaw と限定された視覚補正だけを載せる。これらの小ステップは診断用 CLI としては残るが、`run_open()` の通常経路では使わない。 |
| 13:33–13:46 | 初回の `close-verify` → `proof-pull` → 135 点の `open-door`。`...T043325Z_close_verify_demo_contact/`、`...T043548Z_proof_pull_demo_contact/`、`...T043620Z_open_door_demo_contact/` | closed aperture は `0.4123`、5 mm 後は `0.3663` で proof は通った。しかし full pull の終端 aperture は `0.0034` で、端点画像も開扉を支持しなかった。つまり短い proof だけでは長い回転 pull の把持を保証しない。 | 一発で 135 点を流す方式を廃止。2 倍時間スケール、15 frame ごとの aperture checkpoint、空把持時の即停止・15 mm 退避を実装した。 |
| 13:52–14:06 | 画像整合後の再把持と checkpoint pull。`...T044420Z_retry2_*`、`...T045946Z_retry4_*`、`...T050457Z_retry5_*` | retry 2 / 4 / 5 はそれぞれ close/proof で `0.407/0.365`、`0.323/0.319`、`0.293/0.291` と非空だったが、いずれもおおむね demo frame 171 で滑脱した。各 `*_open_slow_checkpointed/` は `before/` だけを残して終了しており、checkpoint による中断と一致する。 | aperture が残ることを「正しい recessed handle を挟んだ」証拠にしない。赤いラベルの無制限な反復補正、開始姿勢からの増分補正、直接 joint 微調整は通常経路に採用しなかった。後者は会話ログでも関節誤差 `0.178 rad` へ悪化して停止している。 |
| 14:11–14:48 | 8 枚の head RGB-D から扉面を推定し、yaw 補正後に再把持。`incubator_door_20260808_retry6_yaw_aligned_*` | ユーザーが 13:34 に「黒い爪状部分は影」と明示したため、黒領域を根拠から除外した。平面の実測は約 `-5.33°`（後に基準 `-5.38°` として固定）。最終試行は proof まで把持を保ち、full pull 中に滑ったが、head 画像で開扉を確認して退避・開爪した。 | `2b6ccab` に残った核は、RGB-D 平面 yaw、影非依存の bounded visual alignment、fresh contact 姿勢を基準にした closure / proof / checkpoint である。なおこの時点の「開いた」は画像確認であり、次の自律 run の登録済み RGB-D endpoint 判定ほど強い証拠ではない。 |
| 17:29–17:52 | 開扉軌道を逆走する閉扉と、Peacock の専用 close demo の比較。`incubator_door_close_20260808T083305Z_reverse_open/`、`...T084010Z_peacock_demo/`、`...T085056Z_peacock_raw/` | 逆走は開扉把持高さが約 15 cm 高く、扉を押せなかった。回転補正なしの raw `door_close_20260703_163736` は、開爪の低い押し軌道として閉扉した。 | `run_close()` は `close-door-demo` を使う。逆走用の診断 primitive は残っていても、通常の自律閉扉は呼ばない。これは `bfa9b7e` で固定された。 |
| 19:01–19:08 | 新しい `run_incubator_door_autonomy.py open --execute` / `closed --execute` をその場で実行。`incubator_auto_open_20260808_demo*`、`incubator_auto_close_20260808_demo/` | 1 回目は native camera log と JSON の混在でオーケストレーターが結果を読めず停止し、retry は指令なしで停止した。`retry2` は open / close の両 endpoint を確認した。さらに状態 mask に腕が混ざる問題を実機を動かさず修正した。 | process JSON の最外側抽出と、`closed_vertical_plane` と開閉深度差の交差 mask を `f2149bb` / `283b913` に保存した。端点が曖昧なら追加 pull / push を送らない。 |

### 会話から明示的に採用した制約

- 黒い爪状の見え方、影、明度一致を contact / endpoint の根拠にしない。これは会話中のユーザー観察が直接の契機で、RGB-D 平面、機械 aperture、登録済み endpoint に置き換えた。
- `0.41` 程度の閉じ残りと 5 mm proof は有用だが十分条件ではない。失敗した三つの長い pull がこの点を実証したため、full pull の途中にも aperture checkpoint を置いた。
- 赤い EYELA ラベルは recessed handle そのものではなく、剛体親の補助特徴である。照明で成分が欠けた実験があったため、normal flow では最大 3 回・各 15 mm 以下の lateral correction に制限し、検出・収束しなければ contact を拒否する。
- 手書きの増分座標列や直接 joint 微調整を、通常の成功条件にしてはいけない。現在の自律入口は `aligned-yaw-preclose → bounded visual-align → aligned-contact → close → proof → reverify → checkpointed open → recover → endpoint classify` のみを呼ぶ。

したがって、現在も `run_incubator_door_demo.py` に残っている `measured-lateral-step`、`orientation-probe`、逆走 close は「当時の原因切り分けを再現する診断 primitive」であり、通常の `run_incubator_door_autonomy.py` が試行錯誤として繰り返す経路ではない。これはその場のコードを消して成功だけを後付けしたものではなく、失敗を検出するガードと、成功した経路を分離して commit した結果である。

## 実機の成功試行: 閉扉から開扉まで

自動成功ランは `incubator_auto_open_20260808_demo_retry2` である。実際には右カメラの補正は 1 回だけ（局所座標で約 3.5 mm）だった。

| 段階 | 保存された値・結果 |
| --- | --- |
| 初期状態 | `closed`。固定タグ登録残差 0.0077 tag lengths、open 誤差 1.42、closed 誤差 0.28 |
| 扉面 | ライブ yaw -4.61°、基準 -5.38° から +0.775° の残差補正 |
| 接触・閉じ | 安定した非空アパーチャ 0.3057、姿勢ドリフト 0.0° |
| 5 mm proof | アパーチャ 0.2957 を保ち、proof と静止再確認の両方を通過 |
| full pull | 途中 checkpoint で 0.0034 < 0.0739 となり、空把持として停止。盲目的な再 pull はしなかった |
| 終端確認 | 退避後に `open`。固定タグ登録残差 0.0084、open 誤差 0.18、closed 誤差 0.83 |

画像による比較はローカルの次の証拠を参照できる。

- [開始時: closed](../data/runs/pasteur/incubator_auto_open_20260808_demo_retry2/02_state_initial/state_evidence.png)
- [終端: open](../data/runs/pasteur/incubator_auto_open_20260808_demo_retry2/12_state_open_attempt_1_result/state_evidence.png)

終端画像では実際に扉の内側が露出している。従って「途中で jaws が空になったので開扉は失敗」とも、「把持を最後まで保持した」とも言うべきではない。正しくは **proof までは保持し、以後の pull 中に滑ったが、滑る前に扉を開け切り、RGB-D でその端点を確認した** である。

## コードと実験の時系列

| 時点 | 変更／観測 | 因果上の意味 |
| --- | --- | --- |
| 7 月 3 日–8 月 8 日 | 多数のテレオペ開扉を収録し、12 本を成功として選別 | 物理的な handle contact と pull の教師データ |
| 8 月 8 日 14:54 JST, `2b6ccab` | shadow-robust door replay、RGB-D plane、デモコンパイル、右カメラ微小補正を追加 | 絶対 replay と影依存からの脱却 |
| 同日午後 | 位置・高さ・yaw を変えた失敗を含む複数試行 | 空把持や接触ずれのアパーチャ・画像記録を獲得 |
| 8 月 8 日 17:53 JST, `bfa9b7e` | 開扉軌道を逆再生しない、別の open-jaw 閉扉デモを追加 | これは閉扉の改善であり、開扉成功の原因ではない |
| 8 月 8 日 18:13 JST, `f2149bb` | 開/閉状態機械を追加 | 観測→接触→証明→端点確認を一つの実行器にした |
| 8 月 8 日 19:03 JST | `incubator_auto_open_...retry2` が endpoint `open` を確認 | 本書でいう自動開扉の実機証拠 |
| 8 月 8 日 19:08 JST, `283b913` | endpoint evidence を強化 | 成功後の学習を fail-closed な仕様に固定 |
| 8 月 8 日 23:45 JST, `676981b` | cross-lab appliance frame を追加 | 他ラボへの可搬性。元の固定 Pasteur 成功の原因ではない |

## 現在も成立していること、まだ成立していないこと

### 成立している

- door 専用の current source は当時の最終スナップショットから変わっていない。
- 関連回帰テストは 2026-09-04 に `26 passed`（door demo / visual / plane / close / appliance / appliance-frame）を確認した。
- 開扉・閉扉とも、独立した registered RGB-D endpoint で検証する設計になっている。閉扉側の自動ランも `status: success`、closed 誤差 0.018、開扉誤差 1.080 を記録している。

### 未解決または再実証が必要な点

- 取得済みの自動開扉は 1 件の endpoint 成功であり、把持が最後まで持続した成功ではない。再現性や成功率の主張には複数ランが必要である。
- 固定タグ、Record3D のカメラ姿勢、参照端点、インキュベータの取り付けが変われば、古い endpoint reference を使ってはならない。まず read-only enrollment / endpoint capture を更新する。
- 右カメラの赤ラベル補正は小さな平行移動用で、ハンドル検出器ではない。大きな移動・異なる取手形状・遮蔽にはそのまま一般化しない。
- full pull 中の滑りをさらに減らすには、接触姿勢・jaw 力・pull の最初の数 cm の軌道を再評価する必要がある。ただし、停止後に同じ pull を即座に繰り返すのは禁物で、まず端点を観測する現在の方針は維持する。

## 再現・監査の入口

読取り専用の確認は次で行える。

```bash
PY=/home/admin/miniforge3/envs/robot-test/bin/python

$PY -m pytest -q \
  tests/test_incubator_door_demo.py \
  tests/test_incubator_door_visual.py \
  tests/test_incubator_door_plane.py \
  tests/test_incubator_door_close.py \
  tests/test_articulated_appliance.py \
  tests/test_appliance_frame.py \
  tests/test_prepare_appliance_registration.py

jq '.status, .final_state.state' \
  data/runs/pasteur/incubator_auto_open_20260808_demo_retry2/journal.json
```

実機を動かす手順と fail-closed 条件は [PASTEUR_INCUBATOR_DOOR.md](PASTEUR_INCUBATOR_DOOR.md) を正とする。本書はその設計書を置き換えるものではなく、なぜこの構成になったかと、どこまで実証済みかを残す監査用の記録である。
