# RGB-Dから記録軌道までの完全オフライン再実行

## 目的

`src/run_pasteur_offline_replay.py` は、既存のRecord3D、head RGB-D、wrist
TrueDepth、SAM mask、停止時関節状態だけから、次を一つのコマンドで再生成します。

1. 固定head cameraの深度順序を使ったPiper base位置合わせ
2. SAMがarmと顕微鏡を結合したcollision voxelだけの限定的carving
3. 右wrist RGB-Dの青crossを使った最新透明lid位置
4. canonical homeから実測preclose、closed、liftを通るMuJoCo軌道
5. MP4、開始／終了画像、軽量スマホ3D、機械可読report

この入口はrobot clientやRPCをimportせず、`commands_sent=false`です。生成した
軌道は実機実行を許可しません。

## Pasteur回帰データの最短コマンド

```bash
MUJOCO_GL=egl \
/home/admin/miniforge3/envs/robot-test/bin/python \
  src/run_pasteur_offline_replay.py \
  --config src/configs/pasteur_offline_replay_20260730.json \
  --output-dir data/runs/pasteur/offline_replay_20260731_v1
```

初回は全stageを計算します。各stageのcommand、コード、config、capture、
mask、前stage出力をSHA-256に入れます。同じ入力で再実行すると、完全な出力が
存在するstageだけ再利用します。2026-07-30 Pasteurデータでは初回約56秒、
全stage cache hit時は約1.1秒です。数値は実行環境に依存します。

## 座標と左右の不変条件

- production ConeE MJCFだけは歴史的にphysical right = `left_arm_*`、
  physical left = `right_arm_*`
- semantic planning MJCFはphysical right = `right/`、physical left = `left/`
- production branch名をprefix除去してsemantic branch名に変換してはいけない
- 右camera、右controller関節、対象fit、collision carving、trajectoryが
  semantic `right/`で一致しない場合は停止する
- controllerの物理Piper値は原本として保持し、semantic MuJoCoへ入れる時だけ
  `physical_to_semantic_model_q_offset()`の固定joint-zero差を加える
- NYU gripperのhome rollは面のPCAだけで決めない。爪方向の90°曖昧性を解消した
  `semantic_model_home_q()`を使い、左右とも同じjaw-aligned姿勢にする
- homeは `robot.arm.home.physical_home_q()` を唯一の権威とする
- head cameraでbaseを合わせ、wrist cameraで動的対象を合わせる
- 画像左右、SAM instance番号、固定pixel ROIから物理armや対象を決めない

位置合わせstageは派生`positioned_robot.mjcf`のhome keyを毎回
physical left、physical rightの順でsemantic model値へ変換して固定します。
物理home command自体は変更しません。2つのpersistent
baseは古い初期位置との近さでは割り当てず、同期qposで片腕だけが大きく動いた
viewと固定cameraのSAM差分を使って同定します。view名や画像左右は使いません。
この不変条件はrenderer、planner、testでも確認します。

ただしpersistentなSAM robot領域を、そのままbaseとは扱いません。3D中心が
顕微鏡など既知のnon-robot semantic volume内にある候補は除外します。片方だけ
確実に観測できた場合、そのbaseだけを更新し、未観測baseはreview済み位置に
保持します。未観測baseを誤検出物へ移動せず、reportには保持した事実を
明記します。

## 最新の透明対象

透明lidのSAM maskだけでは距離が不安定なので、意味とmetricを分離します。

- SAM/reviewed catalog: 物体がlidであること
- 最大の青領域: gripper/tool基準
- toolに最も近いcross形状の青領域: lid上marker
- aligned TrueDepth: markerの3D点
- 複数の完全停止姿勢: camera-to-EEとepisodeごとの静止対象位置を同時fit
- exact Piper CAD: controller座標からscene座標へ変換
- 支持面: 透明物体中心Zを支持面＋厚さの半分へ拘束

pixel数の固定閾値やROIは使いません。面積、距離、形状は画像寸法で正規化します。
さらに成功episodeで、EE siteと対象の最短距離を対象半径で正規化し、
`target_geometry_gate`で確認します。これにより、画像上は高信頼でも対象が
数十cmずれた誤配置をplannerへ渡しません。

出力:

- `target/wrist_target_report.json`
- `target/latest_target_scene.json`
- `target/overlays/*.png`

## 固定headから現在の皿・蓋を更新

`current_object_refresh`を持つprofileでは、過去wrist target stageの直後に
`src/update_current_semantic_objects.py`を実行します。このstageはaccepted
MuJoCoのstatic geometry、robot base、incubator、microscopeを変更せず、
catalogで指定した可動物体bodyだけを更新します。

```bash
MUJOCO_GL=egl \
/home/admin/miniforge3/envs/robot-test/bin/python \
  src/run_pasteur_offline_replay.py \
  --config src/configs/pasteur_current_scene_20260731.json \
  --output-dir data/runs/pasteur/current_scene_automatic_20260731
```

処理順:

1. 現在head captureの照明・深度品質を確認
2. live SAMまたはprovenance付きaccepted SAM maskから円形instanceを抽出
3. fixed tag bridgeでpixel rayをaccepted scene座標へ変換
4. 透明面のZをsupport plane＋物体厚さの半分へ拘束
5. live SAMが同一promptで複数instanceを返した場合、画像左右ではなく、前回
   accepted modelからの移動量を物体半径で正規化して割り当て
6. best/second-best assignmentのmarginが小さい場合は停止
7. 派生MuJoCoの対象bodyだけ更新し、NYU gripper、jaw-aligned home、左右identity、
   body座標をcompile後に再検証
8. 更新済みmodel/object sceneを軌道、動画、mobile 3Dへ渡す

保存済みmaskで再現する場合はprofileの`accepted_masks`を使います。新しいlive
captureでは`accepted_masks`を省き、`--sam-endpoint tcp://HOST:PORT`を指定します。
固定pixel ROIやimage-left/right規則は使いません。

出力:

- `current_scene/scene.mjcf`
- `current_scene/latest_target_scene.json`
- `current_scene/current_object_report.json`
- `current_scene/current_objects_overlay.png`

## 現在の蓋に対する物理把持探索

`simulated_grasp_search.enabled=true`では、現在対象を反映した直後に
`src/optimize_lid_grasp_trajectory.py`を実行します。承認済みNYU gripperの外観と
姿勢は基準モデルに残し、派生したsimulation-only MJCFでのみ左右可動padと
lid free jointを追加します。探索順はhome、上空XY、降下、水平挿入、閉じる、
verification lift、holdです。liftは一発のjoint-space補間ではなく、対象poseの
XYとgripper姿勢を固定した8個のCartesian Z waypointで40 mm上げます。

候補は、hold中の両pad接触、物体による閉じ残り、20 mm以上の持上げ、
lift/hold中のlid/grasp相対距離が物体半径の1.5倍以下、gripperのXY逸脱が
物体半径の7.5%以下、対象座標系内での把持点滑りが物体半径の10%以下を
全て満たす場合だけ成功です。粗いsupport voxelは完成済み対象と二重衝突
しないよう局所接触から外し、対象底面から生成した滑らかな作業台と皿を
支持authorityにします。出力は
`grasp_search/`の動画、最終画像、全候補report、物理検証済み最良軌道JSONです。
閉じ量はreplay profileの静止済み`closed_nonempty`実測captureから読み、
その`right_gripper_open_ratio`をclose、lift、hold knotへ保存します。実機の
リンク機構を未校正のproxy half-gapへ線形変換してはいけません。
これは把持幾何のsimulation検証であり、元sceneのESDF接触やcamera-to-robot
authorityを上書きせず、実機motion authorityも付与しません。

## 記録軌道の意味

連続teleop関節ログは保存されていません。したがって虚偽の「元動画と同一軌道」
とは記録しません。

正確に保存されているもの:

- canonical home
- success preclose停止時のright qとgripper
- closed non-empty停止時のright qとgripper
- verification lift停止時のright qとgripper

点間は次の順で再構成します。

1. 直線joint edgeが衝突なしなら採用
2. 衝突する場合は複数の決定的seedでbidirectional RRT-Connect
3. shortcut後のjoint path長が最短の候補を採用
4. 各edgeをquintic minimum-jerkで時間化
5. 全sampleでmoving physical-right/semantic-rightのMuJoCo contactを再確認
6. 4つのendpointが入力値と誤差0であることを確認

`trajectory/trajectory.json` は時刻、物理q、model q、前sampleとの差分、
gripper比、stage、実測endpoint、planner knot、全contact auditを持ちます。

## SAM衝突汚染の限定補正

armが顕微鏡と画像上で接すると、SAM maskが一体化し、顕微鏡のcollision voxelに
arm表面が混入します。carvingは次をすべて満たすvoxelだけを削除します。

- allowlistした意味body prefix
- accepted depth-persistence alignment
- canonical homeまたは明示した実測済み停止姿勢でexact robot CADと重なる
- 削除割合がconfig上限以下

観測visual mesh、支持面、box template、allowlist外の物体は変更しません。
moving physical-right armの接触が残る場合は、支持面や対象物を削って通しては
いけません。出力をdisplay-onlyに保ち、base pose、end-effector collision、
支持面、動的対象の順に再較正します。静止側を含むglobal sceneに接触があれば
`global_scene_home_clear=false`、`hardware_motion_authorized=false`です。

## 出力とスマホ表示

```text
OUTPUT/
  alignment/
  collision_scene/
  target/
  current_scene/
  trajectory/trajectory.json
  render/recorded_replay.mp4
  render/recorded_replay_start.png
  render/recorded_replay_final.png
  mobile/
  pipeline_report.json
  index.html
  cache/
  logs/
```

完全なPlotly HTMLはスマホには大きいため、`src/optimize_plotly_mobile.py`が
表示用の面・点だけを間引き、共通の`plotly.min.js`を一度だけ配信します。
元のengineering mesh、MuJoCo collision、ESDFは変更しません。

```bash
systemd-run --user --unit=piper-semantic-viewer \
  --property=Restart=always \
  /usr/bin/python3 -m http.server 8784 --bind 0.0.0.0 \
  --directory /absolute/path/to/OUTPUT
```

Tailscaleでは `http://TAILSCALE_IP:8784/` を開きます。

## 別環境への一般化

コードへ新しい座標やpx閾値を足さず、profileを追加します。

- static scene/SAM catalog: semantic scene profile
- head capture、mask、support height: offline replay profile
- wrist色、dimensionless形状gate、capture episode: wrist target profile
- 実測endpoint、速度、planner seeds: replay profile
- 物体寸法: catalogまたはobject profile

新しい透明対象に色markerがない場合は、SAM identityに加え、depth-validな別の
metric anchorまたは多視点shape fitが必要です。SAM maskだけから透明縁の正確な
3Dを推測してmotion authorityへ昇格させません。
