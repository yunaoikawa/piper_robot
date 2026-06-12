# piper-mujoco

Piper single-arm ロボットを MuJoCo でシミュレートし、フラスコを掴んで持ち上げるタスクを強化学習で訓練するプロジェクト。

実行コマンドはすべて **`piper-mujoco/` ディレクトリ直下** から実行する。

---

## セットアップ

```bash
pip install -r rl/requirements.txt
```

---

## 学習 (`rl/train.py`)

### 基本実行

```bash
python -m rl.train
```

### 設定ファイルを指定

```bash
python -m rl.train --config rl/configs/default.yaml
```

### チェックポイントから再開

```bash
python -m rl.train --resume rl/checkpoints/ckpt_ep000500.pt
```

### 主な設定項目 (`rl/configs/default.yaml`)

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `train.total_episodes` | 10000 | 総エピソード数 |
| `train.log_interval` | 100 | ログ出力間隔 (ep) |
| `train.save_interval` | 500 | チェックポイント保存間隔 (ep) |
| `agent.lr` | 3e-4 | 学習率 |
| `env.max_episode_steps` | 10000 | 1エピソードの最大ステップ数 |

チェックポイントは `rl/checkpoints/` に保存される。学習完了時は `final.pt` として保存される。

### ログの見方

```
[2026-06-12 10:00:00] ep=   100/10000 | step=    51200 | mean_ep_rew= -12.34 | actor= 0.1234 | critic= 0.5678 | entropy=0.9012 | ep/s=12.34 | lifted=False
```

- `lifted` : 直近 100 ep 内でフラスコを持ち上げた成功エピソードが 1 回以上あれば `True`

---

## 評価・可視化 (`rl/eval.py`)

### viewer なし（SSH 先など、挙動を数値で確認）

```bash
python -m rl.eval --checkpoint rl/checkpoints/final.pt --no_viewer
```

### viewer なし + 軌跡を `.npz` に保存

```bash
python -m rl.eval --checkpoint rl/checkpoints/final.pt --no_viewer --record
```

軌跡は `rl/trajectories/` に `final_ep01.npz` 〜 `final_ep05.npz` として保存される。

### viewer あり（ローカル、または `ssh -X` 経由）

```bash
mjpython -m rl.eval --checkpoint rl/checkpoints/final.pt
```

### オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--checkpoint` | (必須) | 使用するチェックポイントのパス |
| `--n_episodes` | 5 | 実行エピソード数 |
| `--step_delay` | 0.005 | viewer 表示速度 (秒/step) |
| `--record` | off | 軌跡を `.npz` に保存する |
| `--no_viewer` | off | viewer を起動しない (SSH 先推奨) |
| `--save_dir` | `rl/trajectories/` | 軌跡の保存先ディレクトリ |

---

## 軌跡の再生 (`rl/replay.py`)

`eval.py --record` で保存した `.npz` を viewer で再生する。

```bash
mjpython -m rl.replay --traj rl/trajectories/final_ep01.npz
```

### オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--traj` | (必須) | 再生する `.npz` ファイルのパス |
| `--speed` | 1.0 | 再生速度倍率 (2.0=2倍速, 0.5=スロー) |

### キー操作

| キー | 動作 |
|---|---|
| `Space` | 一時停止 / 再生 |
| `R` | 先頭に戻る |
| `→` | 1フレーム進む (一時停止中) |
| `←` | 1フレーム戻る (一時停止中) |

---

## シーンの確認 (`scripts/viewer/view_xml.py`)

XML シーンをシミュレーションしながら viewer で確認する。関節一覧も標準出力に表示される。

```bash
mjpython scripts/viewer/view_xml.py xml/lab-scene.xml
```

ファイル名のみでも `xml/` ディレクトリから自動で探索する。

```bash
mjpython scripts/viewer/view_xml.py lab-scene.xml
```

---

## ディレクトリ構成

```
piper-mujoco/
├── rl/
│   ├── train.py          # 学習エントリポイント
│   ├── eval.py           # 評価・可視化
│   ├── replay.py         # 軌跡の再生
│   ├── configs/
│   │   └── default.yaml  # ハイパーパラメータ設定
│   ├── agent/
│   │   ├── a2c.py        # A2C エージェント
│   │   └── actor_critic.py
│   ├── env/
│   │   └── lab_env.py    # MuJoCo 環境 (フラスコ pick-and-lift)
│   ├── checkpoints/      # 保存されたチェックポイント
│   └── trajectories/     # eval --record で保存した軌跡
├── xml/
│   └── lab-scene.xml     # シーン定義
└── scripts/
    └── viewer/
        └── view_xml.py   # XML ビューワー
```
