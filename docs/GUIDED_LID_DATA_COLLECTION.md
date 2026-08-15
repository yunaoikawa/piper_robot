# ACTなし・ユーザー誘導型の蓋データ収集

このモードは既存のLLM制御およびACT agent collectionから独立している。
推論サーバー、checkpoint、5555–5657番ポートは使わない。共有するのは実機、
ConeE 8081、Record3D、controller lock、データ領域だけである。

## 方式

直近の成功HDF5から成功した把持姿勢と把持後の相対運搬だけを抽出する。
ポリシーが途中で生成した無効なclose/openは再生しない。各試行の順序は固定する。

`現在位置 → 垂直clearance → 水平姿勢化 → hover XY → 一括下降 → close → 垂直lift → REVIEW`

- 下降速度は100 mm/sで、目標深度まで一度に下ろす。
- 下降開始後にXYを動かさない。
- UIの補正は次の試行だけに入る。低姿勢を直接jogしない。
- FAIL時は現在の開度を保持して20 mm垂直退避してから開く。
- 最初はユーザーがSUCCESS/FAILを決める。`AUTO開始`後のみ実測gripperで判定する。
- 成功深度はtask別に固定し、自動的な深さ探索は行わない。

open成功後は保存済み相対運搬を実行して解放し、closeへ移る。close成功後は蓋を
再把持して右へ10 mmずらす。位置は0→右10→20→30→20→10→0→左10→20→30
mmと往復する。repositionは配置生成用であり、既定では学習対象にしない。

## 事前監査

実機、カメラ、controller lockに触れず、基準ファイルと設定だけを検査する。

```bash
cd /home/admin/src/google/robot-vla-data
/home/admin/miniforge3/envs/robot-test/bin/python \
  src/run_guided_lid_collection.py --audit-only
```

`inference_enabled: false`、下降速度`0.1`、両baselineのSHA-256を確認する。

## 起動

ConeEをこのbranchのコードで`--attach-current`起動してからrunnerを起動する。
これにより、関節・FK・右gripper・右torqueを単一snapshotで取得できる。古いConeE
でも同期pressure fallbackは安全に動くが、30 Hz学習適格性を満たさない可能性がある。

```bash
cd /home/admin/src/google/robot-vla-data
/home/admin/miniforge3/envs/robot-test/bin/python \
  src/run_guided_lid_collection.py
```

スマホでは `http://100.127.18.64:8780/` を開く。最初の試行だけ`試行開始`を押す。
以降はSUCCESS後にopen/close/repositionを自動で進め、各把持後のREVIEWで待つ。

補正例:

- 20 mm手前: `手前 X+`、20 mm
- 10 mm右: `右 Y-`、10 mm
- 5 mm深く: `下 Z-`、5 mm

REVIEW中に加えた補正は現在の結果を変えず、FAIL後の次試行に適用される。成功時は
その試行開始時に凍結した補正だけをtask別の成功値として保存する。

## データ

保存先は`data/vla_agent/guided_lid`。

- `success/lid_open`, `success/lid_close`: 完全に終了した成功デモ
- `failures/...`: grasp miss、drop、jamなど
- `pending/...`: 中断・クラッシュ時の未確定データ

学習対象にはSUCCESSに加えて実測30 Hz、dropなし、head/right全サンプル、有効な
実時刻が必要である。ユーザーの待機時間は学習時間軸から除外する。

## 検証

```bash
/home/admin/miniforge3/envs/robot-test/bin/python -m pytest -q \
  tests/test_guided_lid_collection.py \
  tests/test_streaming_gripper_command.py \
  tests/test_agent_collection.py
```
