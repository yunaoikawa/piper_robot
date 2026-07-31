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
- joint値はcontrollerの物理Piper値をそのまま使い、古いMJCF home offsetを
  加えない
- homeは `robot.arm.home.physical_home_q()` を唯一の権威とする
- head cameraでbaseを合わせ、wrist cameraで動的対象を合わせる
- 画像左右、SAM instance番号、固定pixel ROIから物理armや対象を決めない

位置合わせstageは派生`positioned_robot.mjcf`のhome keyを毎回
physical left、physical rightの順でcanonical値へ固定します。2つのpersistent
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
