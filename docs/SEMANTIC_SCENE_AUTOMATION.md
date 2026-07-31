# Codex不要のSAM-first 3D／MuJoCo再構成

## 目的

このフローは、Record3D RGB-DとSAMの意味ラベルから、観測面、補完物体、
支持面、保守的ESDF、Piper CAD、NYU gripperを含むMuJoCoを、Codexを介さず
再現するためのものです。

実行入口は `src/run_semantic_scene_pipeline.py` です。このスクリプトはロボット
RPCをimportせず、実機命令を送りません。見た目が正しくても
`display_ready`、`collision_ready`、`motion_ready` は別々に判定されます。

既存データからhead位置合わせ、最新wrist RGB-D対象、実測grasp keyframe、
MuJoCo軌道、スマホ動画まで再現する入口は
`src/run_pasteur_offline_replay.py`です。詳細は
[OFFLINE_SCENE_REPLAY.md](OFFLINE_SCENE_REPLAY.md)を参照してください。

## 最短の再実行

すでに `multiview_report.json` がある場合:

```bash
MUJOCO_GL=egl python src/run_semantic_scene_pipeline.py \
  --multiview-report data/reconstructions/pasteur/record3d_20260730_215729_v3/multiview_report.json \
  --profile src/configs/pasteur_semantic_scene.json \
  --output-dir data/runs/pasteur/semantic_scene
```

生成物は次の場所にまとまります。

- `OUTPUT/scene/index.html`: スマホ用入口
- `OUTPUT/scene/semantic_3d.html`: 観測面と補完形状
- `OUTPUT/scene/mujoco_home.html`: Piper CADと物体を含むMuJoCo
- `OUTPUT/scene/source_esdf_scene.html`: 元の保守的ESDF
- `OUTPUT/scene/scene.json`: 全物体、支持面、最適化、readiness
- `OUTPUT/pipeline_report.json`: 入力hash、commit、実行command、検証結果
- `OUTPUT/logs/`: 各stageの完全なstdout/stderr

途中stageが失敗しても `pipeline_report.json` を `status=failed` で保存し、
失敗stage、終了code、stderr末尾、完全logへのpathを残します。

## Raw captureから一括実行

SAM3サーバーを起動した状態で、完成したmultiview captureを渡します。

```bash
MUJOCO_GL=egl python src/run_semantic_scene_pipeline.py \
  --capture CAPTURE_DIR \
  --profile src/configs/pasteur_semantic_scene.json \
  --output-dir OUTPUT_DIR \
  --sam-endpoint tcp://127.0.0.1:5562
```

この1コマンドで以下を順番に実行します。

1. 各停止視点を時間中央値でRGB-D化
2. SAMで主要物体をラベル
3. robotと可動物体をstatic registrationから除外
4. Record3D poseとbounded ICPで視点を登録
5. TSDF、意味ボクセル、ポリゴン、保守的ESDFを生成
6. 水平ポリゴンから机と同じ高さの独立した台を抽出
7. SAM物体点群をcatalog形状へ補完
8. 不透明box templateを意味体積とknown-free空間で最適化
9. exact Piper CADへNYU gripperを固定
10. MuJoCoをcompileし、assetとreadinessを検証

SAMを再実行できないoffline replayでは、確認済みmaskを繰り返し指定します。

```bash
python src/run_semantic_scene_pipeline.py \
  --capture CAPTURE_DIR \
  --profile src/configs/pasteur_semantic_scene.json \
  --output-dir OUTPUT_DIR \
  --mask path_00:robot=/absolute/path/robot.png \
  --mask path_00:incubator=/absolute/path/incubator.png
```

## 支持面の自動割当

支持面は画像pxや固定座標で選びません。水平な背景三角形を高さ別に分け、
XYセルのconnected componentで独立した台を作ります。机がarmの遮蔽で
分割されても、最低面のcomponentは同じbenchとしてまとめます。ただし
衝突セルは結合しないため、穴や切り欠きは残ります。

Pasteur固有の「皿と蓋は手前の台、incubatorは奥の台」という関係は
`src/configs/pasteur_semantic_scene.json` の `support_assignment` にあります。
Pythonには物体名や座標をハードコードしていません。

```json
{
  "depth_axis": 1,
  "depth_sign": 1,
  "semantic_roles": {
    "incubator": "rear_elevated",
    "petri_dish": "front_elevated",
    "petri_lid": "front_elevated"
  }
}
```

別環境ではaxis/sign/roleをprofileだけで変更します。

## 体積最適化

PCAは欠けた面の点数に引っ張られるため、opaqueなbox templateの最終姿勢には
使いません。最初の候補だけをPCAで作り、重力方向と物体寸法、支持面高さを
固定した上で `center_x`、`center_y`、`yaw` を最適化します。

目的関数は:

```text
L =
  w_surface * (Q65(|distance(observed_surface, box_surface)|)
               + 0.4 * Q90(...))
  + w_semantic * fraction(semantic_voxels outside box shell)
  + w_free * volume(box ∩ observed_known_free) / volume(box)
```

重要な点:

- `known free` はRGB-Dで明示的に観測されたfree voxelだけ
- unknown空間はfreeとみなさず、損失を加えない
- Z-up、支持面高さ、catalog寸法は固定
- search範囲、sampling数、weight、seedはprofileに保存
- 最適化前後のloss、free侵入率、yaw、改善率を`scene.json`へ保存
- 改善率がprofile閾値未満なら最適化結果を採用しない
- eligibleなboxで最適化が未実行／不採用ならpipeline検証は失敗

現在のPasteur incubator回帰データでは:

```text
objective:              0.3524 -> 0.0722
known-free intrusion:   51.29% -> 0.63%
improvement:            79.5%
yaw:                    -23.9° -> 15.5°
```

数値は入力データから毎回再計算され、固定poseとしてコードに保存されません。

## 日次確認とcollision昇格

初回実行は `daily_scene.json` を `pending_confirmation` として作ります。
ユーザーが物体の有無、ラベル、形状を確認した後:

```bash
python src/run_semantic_scene_pipeline.py \
  --multiview-report MULTIVIEW_REPORT \
  --profile src/configs/pasteur_semantic_scene.json \
  --output-dir OUTPUT_DIR \
  --daily-scene OUTPUT_DIR/scene/daily_scene.json \
  --resume-confirmed
```

`--require-collision-ready` を付けると、以下が揃わない限り非0で終了します。

- accepted camera-to-robot calibration
- synchronized 12-joint qpos
- operator-confirmed daily objects
- MuJoCo compile成功
- NYU gripper必須geomあり、stock link7/link8なし
- robot/environment penetrationが設定上限以下

motion readinessはこのpipelineでは付与しません。経路のswept-volume検証は
独立した下流stageです。

## タグレスcamera-to-robot calibration

head cameraを固定し、人が5姿勢以上へテレオペして完全に停止させます。

```bash
python src/capture_record3d_multiview.py \
  --operator-action move-robot --robot-state \
  --view baseline --view left_excitation --view right_excitation \
  --view both --view holdout

python src/calibrate_head_robot_from_cad.py \
  --capture CAPTURE_DIR \
  --profile src/configs/pasteur_semantic_scene.json \
  --output CALIBRATION.json
```

最後以外の姿勢でfitし、最後をholdoutにします。各RGB-D burst前後でqposが
変化していた場合はcaptureを拒否します。全姿勢に固定して残るrobot mask領域
は、顕微鏡などの誤認識として除外されます。`accepted=true` のreportだけを
pipelineの `--calibration-report` に渡します。

校正にはprofileの `robot_calibration.model` で指定したproduction MJCFを使い、
scene用の近似robotへfallbackしません。左右Piperは独立したbase transformを持ち、
画面上の左右ではなく、SAM instanceの時間追跡と左右別qpos変化から対応付けます。
ConeE production MJCFのcontroller/branch入れ替えは
`robot_calibration.physical_to_production_branch` に明記します。一方、scene用
semantic MJCFは `semantic_robot.physical_to_semantic_branch` で物理左右を保持
します。productionの `left_arm_*` をprefix除去してsemantic `left/` と解釈する
ことは禁止です。

fixed-head refinementではpersistentなrobot-mask componentもsemantic sceneと
照合します。3D中心が既知の顕微鏡などnon-robot物体の完成体積内にあれば、腕の
遮蔽でdepth差分が出ていてもbase候補から除外します。片側しか確実に見えない
場合は、そのbaseだけを更新し、もう片側はreview済み位置のまま
`unobserved` としてreportに残します。

RGB、depth、mask、内部パラメータはcapture時の同一sensor座標のまま計算します。
スマホやaudit画像を見やすく回転する場合も表示専用とし、校正入力の一部だけを
回転させてはいけません。

## スマホ配信

一時確認:

```bash
python src/run_semantic_scene_pipeline.py \
  --multiview-report MULTIVIEW_REPORT \
  --profile PROFILE \
  --output-dir OUTPUT_DIR \
  --serve --bind 0.0.0.0 --port 8784
```

会話やshell終了後も残すLinux user service:

```bash
systemd-run --user --unit=piper-semantic-viewer \
  --property=Restart=always --property=RestartSec=2 \
  /usr/bin/python3 -m http.server 8784 --bind 0.0.0.0 \
  --directory /absolute/path/to/OUTPUT_DIR
```

Tailscale上では `http://TAILSCALE_IP:8784/` を開きます。

## 回帰テスト

```bash
PYTHONPATH=. python tests/test_multiview_semantic_completion.py
PYTHONPATH=. python tests/test_multiview_scene.py
PYTHONPATH=. python tests/test_reconstruct_multiview_scene.py
PYTHONPATH=. python tests/test_build_semantic_scene.py
```

テストには、支持面のfront/rear割当、穴保持、2本のarm分離、同期qpos gate、
6DoF CAD calibration、画面左右に依存しないarm追跡、tool anchorの剛体offset、
known-freeを避ける体積最適化が含まれます。
