# Piper Bimanual Robot — Teleop, Training & Inference Pipeline

End-to-end pipeline for bimanual robot manipulation using Piper arms, VR teleoperation, and Pi0.5 policy learning.

## Overview

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
| Gripper switch | Omron D2F-01F micro switch → Arduino Nano (serial, /dev/ttyUSB0) |
| Cameras | 2× iPhone via Record3D USB (device 0 = head, device 1 = right wrist) |
| VR Controller | Meta Quest → ZMQ (tcp://192.168.1.36:5555) |
| Robot PC | pasteur (Ubuntu, conda env: robot-test) |
| GPU Server | peacock05 via peacock.tsudalab.org (SSH jump host) |

## Repository Structure

```
piper_robot/
├── robot/                          # Robot server (runs on pasteur)
│   ├── cone_e.py                   # Main RPC server with gripper switch
│   ├── arm/
│   │   └── arm.py                  # ArmNode with URDF swap + Dynamixel
│   ├── cone-e-description/         # MJCF models and meshes
│   ├── rpc.py                      # RPCClient/RPCServer
│   └── teleop/
│       └── oculus_msgs.py          # VR controller message parser
│
├── rollout/                        # Inference controller (runs on pasteur)
│   ├── controller.py               # PolicyController with wrist camera
│   ├── camera.py                   # Head + USB wrist camera managers
│   ├── recorder.py                 # HDF5 + MP4 episode saver
│   ├── episode.py                  # Episode lifecycle & autonomous mode
│   ├── keyboard.py                 # Keyboard controls (s/e/h/q)
│   └── manipulability.py          # Jacobian-based manipulability
│
├── cloud_inference/                # Inference server (runs on peacock05)
│   ├── hpc_inference_pi05.py       # ZMQ inference server
│   └── pi05_inference.py           # Pi0.5 policy wrapper
│
├── src/                            # Data processing & evaluation
│   ├── convert_to_lerobot.py       # HDF5+MP4 → LeRobot v3.0
│   ├── eval_offline.py             # Offline policy evaluation
│   └── finetune_pi05.sbatch        # Training script (SLURM/bash)
│
├── teleop.py                       # VR teleoperation data collector
├── cloud_inference_control_collect_v2.py  # Inference client
├── gripper_switch_monitor.py       # Arduino switch debug tool
└── README.md
```

## Setup

### pasteur (Robot PC)

```bash
conda activate robot-test

# CAN bus (run after every reboot, or add to .bashrc)
sudo ip link set can_left down && sudo ip link set can_right down
sudo ip link set can_left type can bitrate 1000000
sudo ip link set can_right type can bitrate 1000000
sudo ip link set can_left up && sudo ip link set can_right up

# Arduino driver (first time only)
sudo modprobe ch341
```

### peacock05 (GPU Server)

```bash
# SSH config (~/.ssh/config on Mac and pasteur)
Host peacock05
    HostName peacock05
    User yoikawa
    ProxyJump yoikawa@peacock.tsudalab.org
    ServerAliveInterval 60
    ServerAliveCountMax 120

# Auto-activates .venv and cd ~/src/robot on login (add to ~/.bashrc)
```

## 1. Data Collection (Teleop)

### Start robot server
```bash
# Terminal 1 (pasteur)
python -m robot.cone_e --switch-port /dev/ttyUSB0
```

### Start teleop
```bash
# Terminal 2 (pasteur)
python teleop.py --switch-port /dev/ttyUSB0
```

### Controls
| Button | Action |
|--------|--------|
| Right A | Start right arm teleop |
| Right B | Stop right arm teleop |
| Left X | Start left arm teleop |
| Left Y | Stop left arm teleop |
| Right trigger | Close gripper |
| ANY start | Begin recording |
| ALL stop | Save episode + home |

### Output format
```
teleop_demonstrations/
├── type_even/        # Even episodes (e.g., put_in task)
│   ├── episode_0000_20260316_*.hdf5
│   ├── episode_0000_20260316_*_head.mp4
│   ├── episode_0000_20260316_*_left.mp4
│   └── episode_0000_20260316_*_right.mp4
└── type_odd/         # Odd episodes (e.g., take_out task)
```

### HDF5 fields
| Key | Shape | Description |
|-----|-------|-------------|
| timestamps | (T,) | Unix timestamps |
| left_ee_pos | (T, 3) | Left EE position |
| left_ee_quat | (T, 4) | Left EE quaternion (wxyz) |
| left_gripper | (T,) | Left gripper state |
| right_ee_pos | (T, 3) | Right EE position |
| right_ee_quat | (T, 4) | Right EE quaternion (wxyz) |
| right_gripper | (T,) | Right gripper state (from switch) |
| depth_frames | (T, H, W) | Depth frames (gzip) |

### Tips for good demos
- **Pause 2 seconds** between phases (reach → pause → grip → pause → lift → pause → place → pause → release → pause → retract)
- Slow, deliberate movements improve learning
- Gripper switch records actual grip state, not VR trigger

## 2. Data Conversion

```bash
# On peacock05
python src/convert_to_lerobot.py \
    --data_dirs data/flask/put_in data/flask/take_out \
    --task_names "put the flask in the incubator" "take the flask out of the incubator" \
    --output_dir data/train \
    --repo_id yoikawa/flask_tasks \
    --fps 30
```

### State/Action format (20D)
```
[left_pos(3), left_r6(6), left_gripper(1), right_pos(3), right_r6(6), right_gripper(1)]
```
Actions = state[t+1] (next-state behavioral cloning).

## 3. Training

```bash
# On peacock05 — always the same command
bash src/finetune_pi05.sbatch
```

- First run: trains from `lerobot/pi05_base`
- Subsequent runs: auto-resumes from `outputs/main/checkpoints/last/pretrained_model`
- Checkpoint always at: `outputs/main/checkpoints/last/pretrained_model`

### Key parameters (edit in finetune_pi05.sbatch)
| Parameter | Default | Notes |
|-----------|---------|-------|
| STEPS | 50000 | ~10 epochs for 147 episodes |
| BATCH_SIZE | 8 | Lower (4) may help with sequential precision |
| compile_model | false | true causes CUDA graph errors |

## 4. Inference

### Start inference server (peacock05)
```bash
# Use tmux to persist across SSH disconnects
tmux new -s inference

python cloud_inference/hpc_inference_pi05.py \
    --checkpoint outputs/main/checkpoints/last/pretrained_model \
    --obs-port 8555 \
    --action-port 5556 \
    --device cuda \
    --pred_horizon 50

# Detach: Ctrl+B → D
# Reattach: tmux attach -t inference
```

### SSH tunnel (pasteur → peacock05)
```bash
bash ~/tunnel_peacock.sh
# Or manually:
ssh -f -N \
    -o ServerAliveInterval=60 \
    -L 15555:localhost:8555 \
    -L 15556:localhost:5556 \
    -L 15557:localhost:5557 \
    peacock05
```

### Start controller (pasteur)
```bash
python cloud_inference_control_collect_v2.py \
    --record \
    --host localhost \
    --obs-port 15555 \
    --action-port 15556 \
    --task "put the flask in the incubator" \
    --episode-timeout 300 \
    --rate 30
```

### Controls
| Key | Action |
|-----|--------|
| s | Start episode (apply actions to robot) |
| e | End episode |
| h | Toggle home/rest position |
| q | Quit |

## 5. Offline Evaluation

```bash
# On peacock05
python src/eval_offline.py \
    --checkpoint outputs/main/checkpoints/last/pretrained_model \
    --data_dirs data/flask/put_in data/flask/take_out \
    --task_names "put the flask in the incubator" "take the flask out of the incubator" \
    --dataset_root data/train \
    --repo_id yoikawa/flask_tasks \
    --num_episodes 5
```

### Interpreting results
| Metric | Good | Bad |
|--------|------|-----|
| left/right_pos | < 0.02m | > 0.05m |
| left/right_rot | < 0.1 | > 0.5 |
| gripper | < 0.1 | > 0.5 |

## Hardware Configuration Notes

### CAN Bus Swap
Physical arms are swapped. `setup_can_buses.sh` maps USB ports:
- `1-1:1.0` → `can_left` (physically right arm)
- `1-6:1.0` → `can_right` (physically left arm)

### arm.py URDF Swap
To compensate for physical swap:
- `is_left_arm=True` → uses `piper_description_right.xml`
- `is_left_arm=False` → uses `piper_description_left.xml`

### Gripper (Dynamixel)
- Port: `/dev/ttyUSB1`, Baudrate: 115200, ID: 1
- Position Mode 3, DXL_POS_OPEN=2800, DXL_POS_CLOSE=0
- Profile Velocity=0 (max speed)
- Only on RIGHT arm (`use_gripper=False` for left)

### Gripper Switch (Arduino Nano)
- Port: `/dev/ttyUSB0`, Baudrate: 115200
- Pin D2 with internal pullup, sends '1' (gripping) or '0' (open) at 100Hz
- Used in `cone_e.py` for inference observation
- Used in `teleop.py` for recording actual grip state

### Cameras (Record3D)
- Requires NYU fork: `github.com/NYU-robot-learning/anysense-streaming`
- numpy must be 1.26.4 (2.x incompatible)
- `np.array()` required to prevent static frames
- Device 0 = head camera, Device 1 = right wrist camera

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Failed to transmit CAN frame` | Re-init CAN buses (see Setup) |
| `buffer_empty` on pasteur | Check SSH tunnel and inference server |
| `ZeroDivisionError` in inference | Camera not connected or returning empty frames |
| `compile_model` CUDA error | Set `"compile_model": false` in checkpoint config.json |
| Gripper oscillating | Check switch wiring or use hysteresis in `get_right_gripper_exact()` |
| Controller moves wrong arm | Verify CAN/URDF/teleop swap chain |
| Tunnel disconnected | Run `bash ~/tunnel_peacock.sh` |

## Known Limitations

- **Left wrist camera**: Not available during inference (black image sent). Train with 2 cameras or add 3rd iPhone.
- **Transparent objects**: VLA struggles with transparent flasks. Consider depth fusion or opaque markers.
- **Action simultaneity**: "release + retract" may happen at same time. Add 2-second pauses in demos between phases.

## Citation

If you use this pipeline, please cite:
```
@misc{oikawa2026piper,
  title={Piper Bimanual Robot Teleop and Pi0.5 Fine-tuning Pipeline},
  author={Oikawa, Yuna},
  year={2026},
  url={https://github.com/yunaoikawa/piper_robot}
}
```