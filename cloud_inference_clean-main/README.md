# Run Your Policy

## Rolling Out the Policy

> **Note:** This repo only contains the pipeline to be run on the cluster. You need to make appropriate changes to the code to use your own policy model as detailed below.

### Step 1: Start Policy Inference on the server

Run the sbatch scripts on the server:

```bash
bash inference_server_pi05_multiview.sh
```

> **Note:** For pi0.5 policy, the model needs some time to be compiled by torch. If the compilation time is too long, change the `compile_mode` field in `config.json` and `train_config.json` in the chekpoint directory. (See [here](https://docs.pytorch.org/docs/stable/generated/torch.compile.html))

### Step 2: Run the Workstation Controller

On the workstation, run:

```bash
conda activate piper && cd robot
python cloud_inference_control_collect_v2.py --record --host localhost --obs-port 15555 --action-port 15556
```

Add the `--autonomous` flag for autonomous mode:

```bash
conda activate piper && cd robot
python cloud_inference_control_collect_v2.py --record --host localhost --obs-port 15555 --action-port 15556 --autonomous
```

---

## HPC Inference Script

> **Note:** Change the command line arguments in the `inference_server.sh` script to use the path to your own policy checkpoint.

---

## HPC Policy Runner

### Observation Format

The observation is sent from the workstation in the following format:

```python
observation = {
    'qpos': np.concatenate([
        left_pose,
        np.array([left_gripper_binary], dtype=float),
        right_pose,
        np.array([right_gripper_binary], dtype=float),
    ]),
    "images": {
        "cam_high": rgb_frame,
        "cam_left_wrist": left_wrist_frame,
        "cam_right_wrist": right_wrist_frame,
    },
    "timestamp": timestamp,
    "rgb_timestamp": rgb_timestamp,
}
```

### Action Format

The expected action format from `policy.forward()` output is of shape `(T, 16)`.

The 16 dimensions are ordered as follows:

| Dimensions | Description |
|------------|-------------|
| 1–7 | Left EEF delta pose (wxyz-xyz quaternion format) |
| 8–14 | Right EEF delta pose (wxyz-xyz quaternion format) |
| 15 | Left gripper **state** (no delta) |
| 16 | Right gripper **state** (no delta) |

> **Important:** The rotation is the *time-difference relative rotation*, **not** simple subtraction.

> **Axis orientation for real robot:** The x-axis is the front-facing direction for the robot. The y-axis is towards the left. The z-axis is upwards.

### Chunked Actions

Previously, only `T=1` was tested. The code may need slight modifications for executing chunked actions. There is currently no execution strategy implemented for handling overlapping chunks or open-loop execution.

---

## Implementation Requirements

You need to implement your own policy wrapper with the following:

1. **`forward()` method**  — Takes in the observation and outputs the action in the format described above. This is already implemented in `pi05_inference.py`.
2. **`load_policy_model()` method** in `hpc_inference_pi05.py` — Ensure the policy is loaded correctly and returns a `Policy` object conforming to the expected formats.

---

## Workstation Controller Usage

### Autonomous Mode (`--autonomous` flag)

- After a 5-second countdown + 2-second pause, the policy starts automatically.
- It resets, saves data, and starts new episodes automatically.
- **Manual override:** Press **`e`** to end the current episode and pause the auto pipeline, then press **`s`** to resume (subsequent episodes will continue automatically).
- **Auto reset conditions** (tentative):
  - 60 seconds after the episode started, **or**
  - Manipulability score drops below threshold (0.005), signaling singularity.

### Manual Mode (without `--autonomous` flag)

1. Press **`s`** after the 5-second countdown to start a rollout episode.
2. While the policy is running, press **`e`** to end the episode.
3. Recorded data and video will be saved at the printed path with a timestamp.
4. After saving, the arms move to the home position.
5. Press **`s`** to start a new episode.

### Keyboard Controls for Both Modes

| Key | Action |
|-----|--------|
| `h` | Toggle between manipulation home position and resting position on the table |
| `s` | Start a new episode (only from home position) |
| `e` | End current episode |

IMPORTANT: the positions are hardcoded in line 87-90 in `arm.py` based on the actual setup. You need to modify the code to use your own positions.

### Ending a Session

When you wish to finish the session:

1. **Option A:** Use teaching mode of the PiPER arms to move them to the resting position.
2. **Option B:** Use **`h`** in the rollout pipeline to move the arms to the resting position.

Once at the resting position, `cone_e.py` can be safely killed, and the arms can be unplugged if needed.

> **Warning:** Starting an episode or resuming auto mode is **not allowed** at the resting position. Use **`h`** to move to the manipulation home position before resuming.
