"""
Convert HDF5+MP4 episode data → LeRobot v3.0 dataset format for Pi0.5 finetuning.

Supports multiple task directories merged into a single dataset,
with optional train/validation split.

Usage:
    # Without validation split
    python convert_to_lerobot.py \
        --data_dirs data/flask/put_in data/flask/take_out \
        --task_names "put the flask in" "take the flask out" \
        --output_dir data/train \
        --repo_id yoikawa/flask_tasks \
        --fps 30

    # With 10% validation split
    python convert_to_lerobot.py \
        --data_dirs data/flask/put_in data/flask/take_out \
        --task_names "put the flask in" "take the flask out" \
        --output_dir data/train \
        --val_output_dir data/val \
        --repo_id yoikawa/flask_tasks \
        --val_repo_id yoikawa/flask_tasks_val \
        --val_split 0.1 \
        --fps 30
"""

import argparse
import json
import os
import glob
from pathlib import Path

import numpy as np
import h5py
import pandas as pd


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------

def quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ], axis=-1).reshape(-1, 3, 3)


def rotmat_to_r6(R: np.ndarray) -> np.ndarray:
    return np.concatenate([R[:, :, 0], R[:, :, 1]], axis=-1)


def quat_wxyz_to_r6(quat: np.ndarray) -> np.ndarray:
    return rotmat_to_r6(quat_wxyz_to_rotmat(quat))


def build_state_r6(pos, quat, gripper) -> np.ndarray:
    r6 = quat_wxyz_to_r6(quat)
    return np.concatenate([pos, r6, gripper[:, None]], axis=-1)


# ---------------------------------------------------------------------------
# Episode loading
# ---------------------------------------------------------------------------

def load_episode(hdf5_path: str) -> dict:
    with h5py.File(hdf5_path, "r") as f:
        left_pos   = f["left_ee_pos"][()]
        left_quat  = f["left_ee_quat"][()]
        left_grip  = f["left_gripper"][()]
        right_pos  = f["right_ee_pos"][()]
        right_quat = f["right_ee_quat"][()]
        right_grip = f["right_gripper"][()]
        timestamps = f["timestamps"][()]

    left_state  = build_state_r6(left_pos,  left_quat,  left_grip)
    right_state = build_state_r6(right_pos, right_quat, right_grip)
    state  = np.concatenate([left_state, right_state], axis=-1).astype(np.float32)
    action = np.concatenate([state[1:], state[-1:]], axis=0)

    return {"state": state, "action": action, "timestamps": timestamps}


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

MP4_SUFFIX_TO_CAMERA = {
    "_head":  "cam_high",
    "_left":  "cam_left_wrist",
    "_right": "cam_right_wrist",
}


def find_episode_pairs(data_dir: str, camera_keys: list) -> list:
    h5_files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    if not h5_files:
        h5_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))

    pairs = []
    for h5 in h5_files:
        stem = Path(h5).stem

        mp4s = {}
        for suffix, cam_key in MP4_SUFFIX_TO_CAMERA.items():
            if cam_key not in camera_keys:
                continue
            mp4_path = Path(data_dir) / f"{stem}{suffix}.mp4"
            if mp4_path.exists():
                mp4s[cam_key] = str(mp4_path)
            else:
                prefix = "_".join(stem.split("_")[:2])
                candidates = sorted(Path(data_dir).glob(f"{prefix}*{suffix}.mp4"))
                mp4s[cam_key] = str(candidates[0]) if candidates else None

        if not any(mp4s.values()):
            single_mp4 = Path(h5).with_suffix(".mp4")
            if single_mp4.exists():
                for ck in camera_keys:
                    mp4s[ck] = str(single_mp4)

        pairs.append((h5, mp4s))
    return pairs


def state_names() -> list:
    names = []
    for side in ["left", "right"]:
        for ax in ["x", "y", "z"]:
            names.append(f"{side}_pos_{ax}")
        for i in range(6):
            names.append(f"{side}_r6_{i}")
        names.append(f"{side}_gripper")
    return names


# ---------------------------------------------------------------------------
# Video re-encoding
# ---------------------------------------------------------------------------

def reencode_video(src_path: str, dest_path: str):
    import av as _av
    import fractions

    raw_frames = []
    with _av.open(str(src_path)) as inc:
        vs = inc.streams.video[0]
        w, h = vs.width, vs.height
        fps_num = int(round(float(vs.average_rate)))
        for frame in inc.decode(video=0):
            raw_frames.append(frame.to_ndarray(format="rgb24"))

    with _av.open(str(dest_path), mode="w", format="mp4") as outc:
        outs = outc.add_stream("libx264", rate=fps_num)
        outs.width   = w
        outs.height  = h
        outs.pix_fmt = "yuv420p"
        outs.options = {"crf": "18", "preset": "fast"}
        for fi, arr in enumerate(raw_frames):
            frame = _av.VideoFrame.from_ndarray(arr, format="rgb24")
            frame = frame.reformat(format="yuv420p")
            frame.pts = fi
            frame.time_base = fractions.Fraction(1, fps_num)
            for pkt in outs.encode(frame):
                outc.mux(pkt)
        for pkt in outs.encode(None):
            outc.mux(pkt)

    return len(raw_frames)


# ---------------------------------------------------------------------------
# Single dataset writer
# ---------------------------------------------------------------------------

def _write_single_dataset(
    episodes: list,
    task_names: list,
    output_dir: str,
    repo_id: str,
    fps: int,
    camera_keys: list,
    label: str = "train",
):
    """Write a single LeRobot v3.0 dataset from a list of episodes."""
    if not episodes:
        print(f"  [{label}] No episodes, skipping")
        return

    root = Path(output_dir)
    data_out    = root / "data" / "chunk-000"
    meta_dir    = root / "meta"
    ep_meta_dir = root / "meta" / "episodes" / "chunk-000"

    data_out.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    ep_meta_dir.mkdir(parents=True, exist_ok=True)

    for ck in camera_keys:
        (root / "videos" / f"observation.images.{ck}" / "chunk-000").mkdir(parents=True, exist_ok=True)

    all_rows = []
    episode_rows = []
    global_frame_idx = 0
    episode_lengths = []

    for ep_idx, (h5_path, mp4s, task_idx) in enumerate(episodes):
        print(f"  [{label}] Episode {ep_idx:04d} (task {task_idx}): {Path(h5_path).name}")

        ep = load_episode(h5_path)
        T = len(ep["state"])
        episode_lengths.append(T)

        rel_ts = (ep["timestamps"] - ep["timestamps"][0]).astype(np.float32)

        for t in range(T):
            row = {
                "episode_index":     ep_idx,
                "frame_index":       t,
                "index":             global_frame_idx + t,
                "timestamp":         float(rel_ts[t]),
                "task_index":        task_idx,
                "next.done":         (t == T - 1),
                "observation.state": ep["state"][t].tolist(),
                "action":            ep["action"][t].tolist(),
            }
            all_rows.append(row)

        global_frame_idx += T

        for ck in camera_keys:
            src_mp4 = mp4s.get(ck)
            dest = root / "videos" / f"observation.images.{ck}" / "chunk-000" / f"file-{ep_idx:06d}.mp4"
            if dest.exists():
                print(f"    {ck}: already exists, skipping")
                continue
            if src_mp4 and Path(src_mp4).exists():
                n_frames = reencode_video(src_mp4, str(dest))
                print(f"    {ck}: re-encoded {n_frames} frames")
            else:
                print(f"    WARNING: No MP4 for {ck}")

        ep_row = {
            "episode_index":      ep_idx,
            "tasks":              [task_idx],
            "length":             T,
            "data_path":          "data/chunk-000/file-000.parquet",
            "dataset_from_index": global_frame_idx - T,
            "dataset_to_index":   global_frame_idx,
            "data/chunk_index":   0,
            "data/file_index":    0,
        }
        ep_start_ts = 0.0
        ep_end_ts   = float(ep["timestamps"][-1] - ep["timestamps"][0])
        for ck in camera_keys:
            vid_key = f"observation.images.{ck}"
            ep_row[f"videos/{vid_key}/chunk_index"]    = 0
            ep_row[f"videos/{vid_key}/file_index"]     = ep_idx
            ep_row[f"videos/{vid_key}/from_timestamp"] = ep_start_ts
            ep_row[f"videos/{vid_key}/to_timestamp"]   = ep_end_ts
        episode_rows.append(ep_row)

    # ── Write parquet ──
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = pd.DataFrame(all_rows)
    schema = pa.schema([
        pa.field("episode_index",     pa.int64()),
        pa.field("frame_index",       pa.int64()),
        pa.field("index",             pa.int64()),
        pa.field("timestamp",         pa.float32()),
        pa.field("task_index",        pa.int64()),
        pa.field("next.done",         pa.bool_()),
        pa.field("observation.state", pa.list_(pa.float32())),
        pa.field("action",            pa.list_(pa.float32())),
    ])
    table = pa.Table.from_pydict(
        {col: pa.array(df[col].tolist(), type=schema.field(col).type) for col in df.columns},
        schema=schema,
    )
    pq.write_table(table, data_out / "file-000.parquet")

    ep_df = pd.DataFrame(episode_rows)
    ep_df.to_parquet(ep_meta_dir / "file-000.parquet", index=False)

    # ── tasks.parquet ──
    unique_task_indices = sorted(set(ti for _, _, ti in episodes))
    tasks_df = pd.DataFrame({
        "task_index": unique_task_indices,
        "task": [task_names[i] for i in unique_task_indices],
    })
    tasks_df = tasks_df.set_index("task")
    tasks_df.to_parquet(meta_dir / "tasks.parquet", index=True)

    # ── info.json ──
    state_dim  = 20
    action_dim = 20
    _names = state_names()

    features = {
        "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
        "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
        "index":         {"dtype": "int64",   "shape": [1], "names": None},
        "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
        "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
        "next.done":     {"dtype": "bool",    "shape": [1], "names": None},
        "observation.state": {"dtype": "float32", "shape": [state_dim], "names": _names},
        "action":            {"dtype": "float32", "shape": [action_dim], "names": _names},
    }
    for ck in camera_keys:
        key = f"observation.images.{ck}"
        features[key] = {
            "dtype": "video",
            "shape": [3, 480, 640],
            "names": ["channel", "height", "width"],
            "path": f"videos/{key}/chunk-{{chunk_index:03d}}/file-{{file_index:06d}}.mp4",
            "video_info": {"fps": fps, "encoding": {"vcodec": "libsvtav1"}},
        }

    info = {
        "codebase_version": "v3.0",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:06d}.mp4",
        "robot_type": "bimanual",
        "total_episodes": len(episodes),
        "total_frames": global_frame_idx,
        "total_tasks": len(task_names),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "repo_id": repo_id,
        "shapes": {
            "observation.state": [state_dim],
            "action": [action_dim],
            **{f"observation.images.{ck}": [3, 480, 640] for ck in camera_keys},
        },
        "names": {"observation.state": _names, "action": _names},
        "features": features,
        "camera_keys": [f"observation.images.{ck}" for ck in camera_keys],
        "video_keys":  [f"observation.images.{ck}" for ck in camera_keys],
        "episode_data_index": {
            "from": [int(x) for x in np.cumsum([0] + episode_lengths[:-1])],
            "to":   [int(x) for x in np.cumsum(episode_lengths)],
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # ── stats.json ──
    state_arr  = np.array([row["observation.state"] for row in all_rows], dtype=np.float32)
    action_arr = np.array([row["action"]            for row in all_rows], dtype=np.float32)
    ts_arr     = np.array([row["timestamp"]         for row in all_rows], dtype=np.float32)

    def _stats(arr):
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std":  arr.std(axis=0).clip(1e-6).tolist(),
            "min":  arr.min(axis=0).tolist(),
            "max":  arr.max(axis=0).tolist(),
            "count": [int(len(arr))],
        }

    img_placeholder = {"mean": [[[0.5]],[[0.5]],[[0.5]]], "std": [[[0.5]],[[0.5]],[[0.5]]],
                        "min": [[[0.0]],[[0.0]],[[0.0]]], "max": [[[1.0]],[[1.0]],[[1.0]]],
                        "count": [int(len(all_rows))]}

    stats = {
        "observation.state": _stats(state_arr),
        "action":            _stats(action_arr),
        "timestamp":         _stats(ts_arr.reshape(-1, 1)),
    }
    for ck in camera_keys:
        stats[f"observation.images.{ck}"] = img_placeholder

    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  [{label}] Dataset ready at: {output_dir}")
    print(f"    Episodes: {len(episodes)}, Frames: {global_frame_idx}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def write_lerobot_dataset(
    data_dirs: list,
    task_names: list,
    output_dir: str,
    repo_id: str,
    fps: int = 30,
    camera_keys: list = None,
    val_split: float = 0.0,
    val_output_dir: str = None,
    val_repo_id: str = None,
    seed: int = 42,
):
    if camera_keys is None:
        camera_keys = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

    assert len(data_dirs) == len(task_names), \
        f"Mismatch: {len(data_dirs)} data_dirs but {len(task_names)} task_names"

    # Collect all episodes
    all_episodes = []
    for task_idx, (data_dir, task_name) in enumerate(zip(data_dirs, task_names)):
        pairs = find_episode_pairs(data_dir, camera_keys)
        if not pairs:
            print(f"WARNING: No episodes found in {data_dir} for task '{task_name}'")
            continue
        print(f"Task {task_idx} '{task_name}': {len(pairs)} episodes in {data_dir}")
        for h5_path, mp4s in pairs:
            all_episodes.append((h5_path, mp4s, task_idx))

    if not all_episodes:
        raise FileNotFoundError("No episodes found in any data directory")

    print(f"\nTotal: {len(all_episodes)} episodes across {len(task_names)} tasks")

    # Split into train/val
    if val_split > 0.0:
        if val_output_dir is None:
            val_output_dir = str(Path(output_dir).parent / "val")
        if val_repo_id is None:
            val_repo_id = repo_id + "_val"

        rng = np.random.default_rng(seed)

        # Stratified split: maintain task distribution
        train_episodes = []
        val_episodes = []

        for task_idx in range(len(task_names)):
            task_eps = [(h5, mp4s, ti) for h5, mp4s, ti in all_episodes if ti == task_idx]
            n_val = max(1, int(len(task_eps) * val_split))
            indices = np.arange(len(task_eps))
            rng.shuffle(indices)
            val_indices = set(indices[:n_val])

            for i, ep in enumerate(task_eps):
                if i in val_indices:
                    val_episodes.append(ep)
                else:
                    train_episodes.append(ep)

        print(f"\nSplit: {len(train_episodes)} train, {len(val_episodes)} val "
              f"({val_split*100:.0f}% val, stratified by task, seed={seed})")

        for task_idx, task_name in enumerate(task_names):
            n_train = sum(1 for _, _, ti in train_episodes if ti == task_idx)
            n_val = sum(1 for _, _, ti in val_episodes if ti == task_idx)
            print(f"  [{task_idx}] '{task_name}': {n_train} train, {n_val} val")

        print(f"\n--- Writing train dataset ---")
        _write_single_dataset(train_episodes, task_names, output_dir, repo_id, fps, camera_keys, "train")

        print(f"\n--- Writing val dataset ---")
        _write_single_dataset(val_episodes, task_names, val_output_dir, val_repo_id, fps, camera_keys, "val")

    else:
        print(f"\n--- Writing dataset (no val split) ---")
        _write_single_dataset(all_episodes, task_names, output_dir, repo_id, fps, camera_keys, "all")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Train: {output_dir}")
    if val_split > 0.0:
        print(f"  Val:   {val_output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs",      nargs="+", required=True)
    parser.add_argument("--task_names",     nargs="+", required=True)
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--repo_id",        required=True)
    parser.add_argument("--fps",            type=int, default=30)
    parser.add_argument("--camera_keys",    nargs="+",
                        default=["cam_high", "cam_left_wrist", "cam_right_wrist"])
    parser.add_argument("--val_split",      type=float, default=0.0,
                        help="Fraction of episodes for validation (0 = no split)")
    parser.add_argument("--val_output_dir", default=None,
                        help="Output dir for val dataset (default: sibling of output_dir)")
    parser.add_argument("--val_repo_id",    default=None,
                        help="Repo ID for val dataset (default: repo_id + '_val')")
    parser.add_argument("--seed",           type=int, default=42,
                        help="Random seed for train/val split")
    args = parser.parse_args()

    assert len(args.data_dirs) == len(args.task_names)

    write_lerobot_dataset(
        data_dirs=args.data_dirs,
        task_names=args.task_names,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        fps=args.fps,
        camera_keys=args.camera_keys,
        val_split=args.val_split,
        val_output_dir=args.val_output_dir,
        val_repo_id=args.val_repo_id,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()