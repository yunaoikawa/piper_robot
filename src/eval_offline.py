#!/usr/bin/env python3
"""
Offline evaluation: feed demo data into trained policy, compare predicted vs actual actions.

Usage:
    python eval_offline.py \
        --checkpoint outputs/pi05_finetune_XXXXXXXX/checkpoints/003000/pretrained_model \
        --data_dirs data/raw/flask/put_in data/raw/flask/take_out \
        --task_names "put the flask in the incubator" "take the flask out of the incubator" \
        --dataset_root data/train \
        --repo_id yoikawa/flask_tasks \
        --num_episodes 5
"""

import argparse
import json
import glob
import os
from pathlib import Path

import numpy as np
import h5py
import torch
import av


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------

def quat_wxyz_to_rotmat(quat):
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return np.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ], axis=-1).reshape(*quat.shape[:-1], 3, 3)


def rotmat_to_r6(R):
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def quat_wxyz_to_r6(quat):
    return rotmat_to_r6(quat_wxyz_to_rotmat(quat))


def build_state_r6(pos, quat, gripper):
    r6 = quat_wxyz_to_r6(quat)
    if gripper.ndim == 1:
        gripper = gripper[..., None]
    return np.concatenate([pos, r6, gripper], axis=-1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_episode_raw(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        data = {
            "left_ee_pos": f["left_ee_pos"][()],
            "left_ee_quat": f["left_ee_quat"][()],
            "left_gripper": f["left_gripper"][()],
            "right_ee_pos": f["right_ee_pos"][()],
            "right_ee_quat": f["right_ee_quat"][()],
            "right_gripper": f["right_gripper"][()],
            "timestamps": f["timestamps"][()],
        }
    left_state = build_state_r6(data["left_ee_pos"], data["left_ee_quat"], data["left_gripper"])
    right_state = build_state_r6(data["right_ee_pos"], data["right_ee_quat"], data["right_gripper"])
    state = np.concatenate([left_state, right_state], axis=-1).astype(np.float32)
    action = np.concatenate([state[1:], state[-1:]], axis=0)
    return state, action, data["timestamps"]


def load_video_frames(mp4_path, max_frames=None):
    frames = []
    with av.open(str(mp4_path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if max_frames and len(frames) >= max_frames:
                break
    return frames


def find_episodes(data_dir):
    h5_files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    episodes = []
    for h5 in h5_files:
        stem = Path(h5).stem
        head_mp4 = Path(data_dir) / f"{stem}_head.mp4"
        left_mp4 = Path(data_dir) / f"{stem}_left.mp4"
        right_mp4 = Path(data_dir) / f"{stem}_right.mp4"
        if not head_mp4.exists():
            single = Path(h5).with_suffix(".mp4")
            if single.exists():
                head_mp4 = single
        episodes.append({
            "hdf5": h5,
            "head_mp4": str(head_mp4) if head_mp4.exists() else None,
            "left_mp4": str(left_mp4) if left_mp4.exists() else None,
            "right_mp4": str(right_mp4) if right_mp4.exists() else None,
        })
    return episodes


# ---------------------------------------------------------------------------
# Policy + preprocessor loading
# ---------------------------------------------------------------------------

def load_policy_and_processors(checkpoint_path, dataset_root, repo_id, device="cuda"):
    """Load PI05 policy and its pre/post-processor pipelines."""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors, make_policy_config
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    checkpoint_path = str(Path(checkpoint_path).resolve())

    # Load policy
    print(f"  Loading policy from: {checkpoint_path}")
    policy = PI05Policy.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.eval()

    # Load config for preprocessor
    config_path = Path(checkpoint_path) / "config.json"
    with open(config_path) as f:
        config_dict = json.load(f)
    policy_type = config_dict.pop("type")
    config = make_policy_config(policy_type, **config_dict)

    # Load dataset stats for normalization
    ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=Path(dataset_root).resolve())
    dataset_stats = {key: {k: torch.tensor(v) for k, v in val.items() if k != "count"}
                     for key, val in ds_meta.stats.items()}

    # Load preprocessor and postprocessor from checkpoint
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=checkpoint_path,
        dataset_stats=dataset_stats,
    )

    return policy, preprocessor, postprocessor


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_episode(policy, preprocessor, postprocessor, episode_info, task_name,
                     device="cuda", subsample=1):
    state, gt_action, timestamps = load_episode_raw(episode_info["hdf5"])
    T = len(state)

    head_frames = None
    if episode_info["head_mp4"]:
        head_frames = load_video_frames(episode_info["head_mp4"], max_frames=T)

    left_frames = None
    if episode_info["left_mp4"]:
        left_frames = load_video_frames(episode_info["left_mp4"], max_frames=T)

    right_frames = None
    if episode_info["right_mp4"]:
        right_frames = load_video_frames(episode_info["right_mp4"], max_frames=T)

    errors = {
        "left_pos": [], "left_rot": [], "left_gripper": [],
        "right_pos": [], "right_rot": [], "right_gripper": [],
        "total": [],
    }

    indices = list(range(0, T - 1, subsample))
    n_eval = 0
    n_errors = 0

    # Reset action queue between episodes
    policy._action_queue.clear()

    for t in indices:
        # Build raw observation (before preprocessing)
        raw_obs = {
            "observation.state": torch.from_numpy(state[t]).unsqueeze(0).float(),
            "task": [task_name],
        }

        if head_frames and t < len(head_frames):
            img = head_frames[t].astype(np.float32) / 255.0
            raw_obs["observation.images.cam_high"] = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

        if left_frames and t < len(left_frames):
            img = left_frames[t].astype(np.float32) / 255.0
            raw_obs["observation.images.cam_left_wrist"] = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

        if right_frames and t < len(right_frames):
            img = right_frames[t].astype(np.float32) / 255.0
            raw_obs["observation.images.cam_right_wrist"] = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            try:
                # Preprocess (normalization + tokenization)
                processed_obs = preprocessor(raw_obs)
                for k, v in processed_obs.items():
                    if isinstance(v, torch.Tensor):
                        processed_obs[k] = v.to(device)

                # Run policy
                pred_raw = policy.select_action(processed_obs)

                # Postprocess (denormalize action back to original scale)
                pred_denorm = postprocessor(pred_raw)
                if isinstance(pred_denorm, dict):
                    pred_action = pred_denorm["action"]
                else:
                    pred_action = pred_denorm

                if isinstance(pred_action, torch.Tensor):
                    pred_action = pred_action.cpu().numpy().flatten()
                else:
                    pred_action = np.array(pred_action).flatten()

            except Exception as e:
                n_errors += 1
                if n_errors <= 3:
                    print(f"    Error at t={t}: {e}")
                continue

        gt = gt_action[t]
        n_eval += 1

        if len(pred_action) >= 20 and len(gt) >= 20:
            errors["left_pos"].append(np.linalg.norm(pred_action[:3] - gt[:3]))
            errors["left_rot"].append(np.linalg.norm(pred_action[3:9] - gt[3:9]))
            errors["left_gripper"].append(abs(pred_action[9] - gt[9]))
            errors["right_pos"].append(np.linalg.norm(pred_action[10:13] - gt[10:13]))
            errors["right_rot"].append(np.linalg.norm(pred_action[13:19] - gt[13:19]))
            errors["right_gripper"].append(abs(pred_action[19] - gt[19]))
            errors["total"].append(np.linalg.norm(pred_action[:20] - gt[:20]))

    print(f"    Evaluated {n_eval}/{len(indices)} frames ({n_errors} errors)")
    return errors


def print_summary(all_errors, label=""):
    print(f"\n{'='*60}")
    print(f"OFFLINE EVALUATION SUMMARY {label}")
    print(f"{'='*60}")

    for key in ["left_pos", "left_rot", "left_gripper", "right_pos", "right_rot", "right_gripper", "total"]:
        vals = all_errors.get(key, [])
        if not vals:
            continue
        arr = np.array(vals)
        print(f"  {key:20s}: mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"median={np.median(arr):.4f}  max={arr.max():.4f}")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Offline policy evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dirs", nargs="+", required=True)
    parser.add_argument("--task_names", nargs="+", required=True)
    parser.add_argument("--dataset_root", default="data/train")
    parser.add_argument("--repo_id", default="yoikawa/flask_tasks")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--subsample", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    assert len(args.data_dirs) == len(args.task_names)

    print(f"Loading policy + processors from {args.checkpoint}...")
    policy, preprocessor, postprocessor = load_policy_and_processors(
        args.checkpoint,
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        device=args.device,
    )
    print("  Ready\n")

    for dir_idx, data_dir in enumerate(args.data_dirs):
        task_name = args.task_names[dir_idx]
        episodes = find_episodes(data_dir)

        if not episodes:
            print(f"No episodes found in {data_dir}")
            continue

        n_eval = min(args.num_episodes, len(episodes))
        print(f"\nTask: '{task_name}' — evaluating {n_eval}/{len(episodes)} episodes")

        task_errors = {k: [] for k in ["left_pos", "left_rot", "left_gripper",
                                        "right_pos", "right_rot", "right_gripper", "total"]}

        for i, ep in enumerate(episodes[:n_eval]):
            print(f"  Episode {i}: {Path(ep['hdf5']).name}")
            ep_errors = evaluate_episode(
                policy, preprocessor, postprocessor, ep, task_name,
                device=args.device,
                subsample=args.subsample,
            )
            for k in task_errors:
                task_errors[k].extend(ep_errors.get(k, []))

        print_summary(task_errors, label=f"[{task_name}]")

    print("Done.")


if __name__ == "__main__":
    main()