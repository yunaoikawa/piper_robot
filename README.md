# Piper Bimanual Robot — Teleop, Training & Inference Pipeline

End-to-end pipeline for bimanual robot manipulation using Piper arms, VR teleoperation, and Pi0.5 policy learning.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        PIPELINE                                 │
│                                                                 │
│  1. Teleop (pasteur)     →  HDF5 + MP4 episodes                │
│  2. Convert (peacock05)  →  LeRobot v3.0 dataset               │
│  3. Train (peacock05)    →  Pi0.5 checkpoint                   │
│  4. Inference (peacock05 + pasteur) → Robot execution           │
└─────────────────────────────────────────────────────────────────┘
```

## Hardware

| Component | Details |
|-----------|---------|
| Arms | 2× Piper 6-DoF (CAN bus, physically swapped → software-compensated) |
| Gripper | NYU string-driven Dynamixel on RIGHT arm only (ID=1, /dev/ttyUSB1) |
| Gripper switch | Omron D2F-01F micro switch → Arduino Nano (/dev/ttyUSB0) |
| Cameras | 2× iPhone via Record3D USB (device 0 = head, device 1 = right wrist) |
| VR Controller | Meta Quest → ZMQ |
| Robot PC | pasteur (Ubuntu, conda env: robot-test) |
| GPU Server | peacock05 via peacock.tsudalab.org (SSH jump host) |

## Repository Structure

```
piper_robot/
├── robot/                              # Robot server (pasteur)
│   ├── cone_e.py                       # RPC server with workspace clamping
│   ├── arm/
│   │   ├── arm.py                      # ArmNode with URDF swap + Dynamixel gripper
│   │   └── urdf/                       # URDF files (left/right swapped)
│   ├── cone-e-description/             # MJCF models and meshes
│   ├── cone_e_mujoco.py                # MuJoCo simulation of ConeE
│   ├── rpc.py                          # RPCClient / RPCServer (ZMQ)
│   ├── base.py                         # Swerve drive base (disabled)
│   ├── lift.py                         # Lift motor control (disabled)
│   ├── test_boundaries.py              # Interactive workspace boundary explorer
│   ├── msgs/                           # Message types (BimanualPose, etc.)
│   └── teleop/
│       ├── oculus_msgs.py              # VR controller message parser
│       ├── oculus_bimanual_teleop.py   # Bimanual VR teleop (direct RPC)
│       ├── oculus_bimanual_node.py     # Bimanual VR teleop (Dora node)
│       ├── oculus_teleop.py            # Single-arm VR teleop
│       ├── oculus_wb_teleop.py         # Whole-body VR teleop
│       └── joystick.py                 # Gamepad base control
│
├── rollout/                            # Inference controller (pasteur)
│   ├── controller.py                   # PolicyController with wrist camera + task param
│   ├── camera.py                       # Head + USB wrist camera managers
│   ├── recorder.py                     # HDF5 + MP4 episode saver
│   ├── episode.py                      # Episode lifecycle & autonomous mode
│   ├── keyboard.py                     # Keyboard controls (s/e/h/q)
│   └── manipulability.py              # Jacobian-based manipulability
│
├── cloud_inference_clean-main/         # Inference server (peacock05)
│   ├── hpc_inference_pi05.py           # ZMQ inference server with action buffer
│   ├── pi05_inference.py               # Pi0.5 policy wrapper (r6→quat conversion)
│   └── inference_server_pi05_multiview.sh
│
├── src/                                # Data processing & training (peacock05)
│   ├── convert_to_lerobot.py           # HDF5+MP4 → LeRobot v3.0 (multi-task, train/val split)
│   ├── train.sh                        # Training script (SLURM/bash, auto-resume)
│   ├── eval_offline.py                 # Offline policy evaluation
│   └── verify_dataset.py              # Dataset verification
│
├── teleop_collect_example.py           # VR teleop data collector (even/odd split)
├── cloud_inference_control_collect_v2.py  # Inference client (imports rollout/)
├── gripper_switch_monitor.py           # Arduino switch debug tool
├── env.yaml                            # Conda environment spec
└── README.md
```

## Setup

### pasteur (Robot PC)

```bash
conda activate robot-test

# CAN bus (run after every reboot)
bash setup_can_buses.sh
# Or manually:
sudo ip link set can_left up type can bitrate 1000000
sudo ip link set can_right up type can bitrate 1000000

# Arduino driver (first time only)
sudo modprobe ch341
```

### peacock05 (GPU Server)

```bash
# SSH config (~/.ssh/config)
Host peacock05
    HostName peacock05
    User yoikawa
    ProxyJump yoikawa@peacock.tsudalab.org
    ServerAliveInterval 60
    ServerAliveCountMax 120

# Activate environment
cd ~/src/robot
source .venv/bin/activate
```

## 1. Data Collection (Teleop)

```bash
# Terminal 1: Start robot server
python -m robot.cone_e

# Terminal 2: Start teleop
python teleop_collect_example.py
# Or with VR relay:
python teleop_collect_example.py --use-relay --relay-host 100.125.255.41
```

### VR Controls
| Button | Action |
|--------|--------|
| Left X | Start left arm teleop |
| Left Y | Stop left arm teleop |
| Right A | Start right arm teleop |
| Right B | Stop right arm teleop |
| Left/Right trigger | Close gripper |
| ANY arm start | Begin recording |
| ALL arms stop | Save episode + home arms |

### Output format
Episodes are split by parity for multi-task collection:
```
teleop_demonstrations/
├── type_even/       # Even episodes (e.g., put_in task)
│   ├── episode_0000_*.hdf5
│   ├── episode_0000_*_head.mp4
│   ├── episode_0000_*_left.mp4
│   └── episode_0000_*_right.mp4
└── type_odd/        # Odd episodes (e.g., take_out task)
```

### Tips for good demos
- Pause 2 seconds between phases (reach → grip → lift → place → release → retract)
- Slow, deliberate movements improve learning
- Action simultaneity (release + retract together) is the main failure mode

## 2. Data Conversion

```bash
# On peacock05
python src/convert_to_lerobot.py \
    --data_dirs data/raw/flask/put_in data/raw/flask/take_out \
    --task_names "put the flask in the incubator" "take the flask out of the incubator" \
    --output_dir data/train/new \
    --val_output_dir data/val/new \
    --repo_id yoikawa/flask_tasks \
    --val_repo_id yoikawa/flask_tasks_val \
    --val_split 0.1
```

### State/Action format (20D)
```
[left_pos(3), left_r6(6), left_gripper(1), right_pos(3), right_r6(6), right_gripper(1)]
```
Actions use next-state behavioral cloning: `action[t] = state[t+1]`.

## 3. Training

```bash
# On peacock05 — always the same command
bash src/train.sh       # direct execution
# or
sbatch src/train.sh     # SLURM submission (if available)
```

The script auto-detects whether to start fresh or resume:
- No checkpoint → trains from `lerobot/pi05_base`
- Existing checkpoint → backs up to `outputs/main_prev_*` and resumes

### Key parameters (edit in src/train.sh)
| Parameter | Default | Notes |
|-----------|---------|-------|
| STEPS | 50000 | Training steps |
| BATCH_SIZE | 8 | Single A100 80GB |
| DATASET_ROOT | data/train/new | Converted LeRobot dataset |
| VAL_ROOT | data/val/new | Validation dataset |
| VAL_FREQ | 500 | Validate every N steps |
| normalization | MEAN_STD | For ACTION and STATE |

Checkpoint always at: `outputs/main/checkpoints/last/pretrained_model`

## 4. Inference

### Start inference server (peacock05)
```bash
tmux new -s inference

python cloud_inference_clean-main/hpc_inference_pi05.py \
    --checkpoint outputs/main/checkpoints/last/pretrained_model \
    --obs-port 8555 \
    --action-port 5556 \
    --device cuda \
    --pred_horizon 50

# Detach: Ctrl+B → D
```

### SSH tunnel (pasteur → peacock05)
```bash
bash ~/tunnel_peacock.sh
# Or manually:
ssh -f -N -L 15555:localhost:8555 -L 15556:localhost:5556 -L 15557:localhost:5557 peacock05
```

### Start controller (pasteur)
```bash
# Terminal 1: Robot server
python -m robot.cone_e

# Terminal 2: Inference client
python cloud_inference_control_collect_v2.py \
    --record \
    --host localhost \
    --obs-port 15555 \
    --action-port 15556 \
    --task "put the flask in the incubator" \
    --episode-timeout 300 \
    --rate 30
```

### Keyboard Controls
| Key | Action |
|-----|--------|
| s | Start episode |
| e | End episode |
| h | Toggle home/rest position |
| q | Quit |

## Hardware Configuration Notes

### CAN Bus Swap
Physical arms are swapped. `setup_can_buses.sh` maps USB ports to compensate:
- `1-1:1.0` → `can_left` (physically right arm)
- `1-6:1.0` → `can_right` (physically left arm)

`arm.py` further swaps URDF files: `is_left_arm=True` → `piper_description_right.xml`.

### Gripper (Dynamixel)
- Port: `/dev/ttyUSB1`, Baudrate: 115200, ID: 1
- Position Mode 3, DXL_POS_OPEN=2800, DXL_POS_CLOSE=0
- Connected directly to pasteur via U2D2 (not through the arm)
- Only on RIGHT arm (`use_gripper=False` for left)

### Cameras (Record3D)
- NYU fork: `github.com/NYU-robot-learning/anysense-streaming`
- numpy 1.26.4 required (2.x incompatible)
- `np.array()` required on every frame to prevent static frames
- Device 0 = head, Device 1 = right wrist
- Left wrist camera not available (black image sent during inference)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Failed to transmit CAN frame` | `sudo ip link set can_left up` (or re-run setup_can_buses.sh) |
| `buffer_empty` on pasteur | Check SSH tunnel and inference server |
| `compile_model` CUDA error | Set `"compile_model": false` in config.json AND train_config.json |
| Gripper oscillating | Policy prediction issue, not sensing |
| Controller moves wrong arm | Verify CAN/URDF swap chain |
| `Communication error` | ZMQ socket corrupted during arm reset; restart controller |
| transformers version error | `sed -i 's/raise ValueError(msg) from None/pass/' modeling_pi05.py` |
| `dataset_stats` kwarg error | `kwargs.pop("dataset_stats", None)` in `from_pretrained()` |
| `gradient_checkpointing` error | Set `--policy.gradient_checkpointing=false` or fix PaliGemma attribute path |

## Known Limitations

- **Left wrist camera**: Not available during inference (black image sent)
- **Transparent objects**: SigLIP struggles with transparent flasks
- **Action simultaneity**: "release + retract" predicted simultaneously at 30Hz; add 2-second pauses in demos
- **Batch size**: Single GPU limited to batch_size=8; collaborator used 128 across multiple GPUs

## LeRobot Modifications

The collaborator's code (`tmp/pi05_training_clean-main/`) patches LeRobot to add validation loss, training-time RTC, and depth support. Applied via `lerobot_replace_helper.py`. Manual fixes required:
1. `modeling_pi05.py` line ~1083: pop `dataset_stats` and `dataset_meta` from kwargs
2. `modeling_pi05.py` line ~569: disable transformer version check
3. `gradient_checkpointing_enable`: attribute path mismatch (`language_model` vs `model.language_model`)