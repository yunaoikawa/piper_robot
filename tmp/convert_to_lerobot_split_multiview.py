#!/usr/bin/env python3
"""
Convert cone_e folding clothes data directly from HDF5+MP4 to LeRobot format.

This script performs a single-step conversion that:
1. Reads HDF5 files for robot states (positions, orientations, grippers)
2. Computes delta actions for training
3. Includes absolute EEF states as observations
4. Properly saves images to the LeRobot dataset
5. Creates separate train and validation datasets

Usage:
    python convert_to_lerobot_split.py \
        --source_dirs shirt_folding_simple_cone_e shirt_folding_simple_cone_e_2 \
        --repo_id username/cone_e_folding_clothes \
        --downsample_factor 4

"""

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from scipy.spatial.transform import Rotation as R
import os


def quaternion_to_6d_rotation(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to 6D rotation representation.

    The 6D representation uses the first two columns of the rotation matrix,
    which provides a continuous representation without gimbal lock.

    Args:
        quat: Quaternion in (w, x, y, z) format (scalar first), shape (4,) or (N, 4)

    Returns:
        6D rotation representation, shape (6,) or (N, 6)
        Format: [r11, r21, r31, r12, r22, r32] (first two columns of rotation matrix)
    """
    # Convert from scalar-first (w, x, y, z) to scalar-last (x, y, z, w) for scipy
    rotation = R.from_quat(quat[..., [1, 2, 3, 0]])

    # Get the rotation matrix (3x3)
    rot_matrix = rotation.as_matrix()  # shape: (..., 3, 3)

    # Extract first two columns and flatten them
    # First column: rot_matrix[..., :, 0], Second column: rot_matrix[..., :, 1]
    rot_6d = np.concatenate([rot_matrix[..., :, 0], rot_matrix[..., :, 1]], axis=-1)

    return rot_6d


def load_episode_from_hdf5(
    hdf5_path: Path,
    video_paths: dict,
    caption_path: Path = None,
    downsample_factor: int = 1
) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load and process a single episode from HDF5 and video files.

    Args:
        hdf5_path: Path to HDF5 file containing robot states
        video_path: Path to MP4 video file
        downsample_factor: Factor to downsample the trajectory

    Returns:
        Tuple of (video_frames, states, actions) where:
        - video_frames: List of RGB frames (H, W, 3) as uint8 numpy arrays
        - depth_frames: Array of shape (T, H, W) with depth data
        - states: Array of shape (T, 20) with absolute EEF states
          [left: x, y, z, r11, r21, r31, r12, r22, r32, gripper,
           right: x, y, z, r11, r21, r31, r12, r22, r32, gripper]
        - actions: Array of shape (T, 20) with delta poses
          [left: dx, dy, dz, dr11, dr21, dr31, dr12, dr22, dr32, gripper,
           right: dx, dy, dz, dr11, dr21, dr31, dr12, dr22, dr32, gripper]
        - captions: List of captions per frame (if caption_path is provided), else None
    """
    # Load video frames for all cameras
    video_frames = {k: [] for k in video_paths.keys()}
    caps = {}
    
    for key, path in video_paths.items():
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
             # Clean up already opened caps
            for c in caps.values():
                c.release()
            raise ValueError(f"Cannot open video: {path}")
        caps[key] = cap

    frame_idx = 0
    try:
        while True:
            # Read from all cameras
            frames_read = {}
            all_ret = True
            
            for key, cap in caps.items():
                ret, frame = cap.read()
                if not ret:
                    all_ret = False
                    break
                frames_read[key] = frame
            
            if not all_ret:
                break

            # Only keep frames at the downsampled rate
            if frame_idx % downsample_factor == 0:
                for key, frame in frames_read.items():
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_frames[key].append(frame_rgb)

            frame_idx += 1
    finally:
        for cap in caps.values():
            cap.release()

    # Load robot states from HDF5
    with h5py.File(hdf5_path, 'r') as f:

        # read raw depth data
        try:
            depth_frames = f['head_depth_frames'][:]  # (T, H, W)
        except:
            depth_frames = f['depth_frames'][:]  # (T, H, W)

        # Some recordings saved depth as (T, W, H) = (T, 256, 192) instead of
        # (T, H, W) = (T, 192, 256).  Rotate each frame 90° CCW to correct.
        if depth_frames.ndim == 3 and depth_frames.shape[1:] == (256, 192):
            depth_frames = np.rot90(depth_frames, k=1, axes=(1, 2))

        # Read left gripper state data
        left_ee_pos = f['left_ee_pos'][:]  # (T, 3)
        left_ee_quat = f['left_ee_quat'][:]  # (T, 4)
        left_gripper = f['left_gripper'][:]  # (T,)
        left_gripper_exact = f['left_gripper_exact'][:]  # (T,)

        # Read right gripper state data
        right_ee_pos = f['right_ee_pos'][:]  # (T, 3)
        right_ee_quat = f['right_ee_quat'][:]  # (T, 4)
        right_gripper = f['right_gripper'][:]  # (T,)
        right_gripper_exact = f['right_gripper_exact'][:]  # (T,)

        # Check raw video count vs raw state count BEFORE downsampling.
        # frame_idx is the total number of raw video frames read by the video loop above.
        raw_state_count = left_ee_pos.shape[0]
        if frame_idx == raw_state_count + 1:
            # Camera captured one extra raw frame at the end.
            # The extra raw frame is at index (frame_idx - 1).  After inline
            # downsampling it is present in video_frames only when
            # (frame_idx - 1) % downsample_factor == 0; in that case trim it.
            if (frame_idx - 1) % downsample_factor == 0:
                print(
                    f"Warning: Raw video count ({frame_idx}) is one more than raw state "
                    f"count ({raw_state_count}) for episode {hdf5_path.stem}. "
                    "Dropping extra video frame (pre-downsampling)."
                )
                for cam_key in video_frames:
                    video_frames[cam_key] = video_frames[cam_key][:-1]
        elif frame_idx != raw_state_count:
            # Large mismatch — will be caught and reported by the post-DS check below.
            pass

        # check that all arrays have the same length
        # T = left_ee_pos.shape[0]
        # assert all(arr.shape[0] == T for arr in [
        #     left_ee_quat, left_gripper, left_gripper_exact,
        #     right_ee_pos, right_ee_quat, right_gripper, right_gripper_exact,
        #     depth_frames
        # ]), f"Mismatched lengths in HDF5 data arrays: {left_ee_pos.shape[0]}, {left_ee_quat.shape[0]}, {left_gripper.shape[0]}, {left_gripper_exact.shape[0]}, {right_ee_pos.shape[0]}, {right_ee_quat.shape[0]}, {right_gripper.shape[0]}, {right_gripper_exact.shape[0]}, {depth_frames.shape[0]}"

        # Downsample if needed
        if downsample_factor > 1:
            depth_frames = depth_frames[::downsample_factor]
            left_ee_pos = left_ee_pos[::downsample_factor]
            left_ee_quat = left_ee_quat[::downsample_factor]
            left_gripper = left_gripper[::downsample_factor]
            left_gripper_exact = left_gripper_exact[::downsample_factor]

            right_ee_pos = right_ee_pos[::downsample_factor]
            right_ee_quat = right_ee_quat[::downsample_factor]
            right_gripper = right_gripper[::downsample_factor]
            right_gripper_exact = right_gripper_exact[::downsample_factor]

        # make sure all arrays have the same length after downsampling
        lengths = [
            depth_frames.shape[0],
            left_ee_pos.shape[0],
            left_ee_quat.shape[0],
            left_gripper.shape[0],
            left_gripper_exact.shape[0],
            right_ee_pos.shape[0],
            right_ee_quat.shape[0],
            right_gripper.shape[0],
            right_gripper_exact.shape[0],
        ]
        min_length = min(lengths)
        if not all(length == min_length for length in lengths):
            print(f"Warning: Mismatched lengths after downsampling. Truncating to minimum length {min_length}. Lengths: {lengths}")
            depth_frames = depth_frames[:min_length]
            left_ee_pos = left_ee_pos[:min_length]
            left_ee_quat = left_ee_quat[:min_length]
            left_gripper = left_gripper[:min_length]
            left_gripper_exact = left_gripper_exact[:min_length]
            right_ee_pos = right_ee_pos[:min_length]
            right_ee_quat = right_ee_quat[:min_length]
            right_gripper = right_gripper[:min_length]
            right_gripper_exact = right_gripper_exact[:min_length]

        # Convert quaternions to rotation objects and 6D rotation representation
        left_rot_obj = R.from_quat(left_ee_quat, scalar_first=True)
        right_rot_obj = R.from_quat(right_ee_quat, scalar_first=True)

        left_rot_6d = quaternion_to_6d_rotation(left_ee_quat)    # (T, 6)
        right_rot_6d = quaternion_to_6d_rotation(right_ee_quat)  # (T, 6)

        # Combine into absolute states for left and right
        left_state = np.concatenate([left_ee_pos, left_rot_6d], axis=1)    # (T, 9)
        right_state = np.concatenate([right_ee_pos, right_rot_6d], axis=1)  # (T, 9)

        # Compute world-frame deltas: delta_rot = next_rot * prev_rot.inv(), delta_pos = next_pos - prev_pos
        T = left_state.shape[0]
        print(f"[DEBUG] Number of timesteps T: {T}", flush=True)

        left_pose_delta = np.zeros_like(left_state)
        right_pose_delta = np.zeros_like(right_state)

        for i in range(T - 1):
            # Left arm
            rel_rot_left = left_rot_obj[i + 1] * left_rot_obj[i].inv()
            rel_pos_left = left_ee_pos[i + 1] - left_ee_pos[i]

            rel_quat_left_scalar_first = rel_rot_left.as_quat(scalar_first=True)
            rel_rot_6d_left = quaternion_to_6d_rotation(rel_quat_left_scalar_first)

            left_pose_delta[i, :3] = rel_pos_left
            left_pose_delta[i, 3:] = rel_rot_6d_left

            # Right arm
            rel_rot_right = right_rot_obj[i + 1] * right_rot_obj[i].inv()
            rel_pos_right = right_ee_pos[i + 1] - right_ee_pos[i]

            rel_quat_right_scalar_first = rel_rot_right.as_quat(scalar_first=True)
            rel_rot_6d_right = quaternion_to_6d_rotation(rel_quat_right_scalar_first)

            right_pose_delta[i, :3] = rel_pos_right
            right_pose_delta[i, 3:] = rel_rot_6d_right

        # Last delta is padded with identity (already initialized)
        left_pose_delta[-1, [3, 4, 5, 6, 7, 8]] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        right_pose_delta[-1, [3, 4, 5, 6, 7, 8]] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        left_gripper_shifted = np.zeros_like(left_gripper_exact[:, None])
        right_gripper_shifted = np.zeros_like(right_gripper_exact[:, None])
        left_gripper_shifted[:-1, 0] = left_gripper_exact[1:]
        left_gripper_shifted[-1, 0] = left_gripper_exact[-1]  # repeat last value for last action
        right_gripper_shifted[:-1, 0] = right_gripper_exact[1:]
        right_gripper_shifted[-1, 0] = right_gripper_exact[-1]  # repeat last value for last action

        # Combine absolute states with gripper values
        # Format: [left_state (9), left_gripper_exact (1), right_state (9), right_gripper_exact (1)]
        states = np.concatenate([
            left_state,
            left_gripper_exact[:, None],
            right_state,
            right_gripper_exact[:, None]
        ], axis=1)  # (T, 20)

        # Combine deltas with absolute gripper states for actions
        # Format: [left_pose_delta (9), left_gripper (1), right_pose_delta (9), right_gripper (1)]
        actions = np.concatenate([
            left_pose_delta,
            left_gripper_shifted,
            right_pose_delta,
            right_gripper_shifted,
        ], axis=1)  # (T, 20)

    # load captions if provided
    if caption_path is not None:
        import json
        print(f"Loading captions from {caption_path}", flush=True)
        with open(caption_path, 'r') as f:
            captions = json.load(f)

        episode_captions = captions.get('labels', '')
        caption_list = [episode_captions[i]["label"] for i in sorted(range(len(episode_captions)), key=lambda i: episode_captions[i]["frame"])]
    else:
        caption_list = None

    # Verify frame count matches state count for all cameras
    # Use 'head' as reference, assuming all videos are same length
    ref_cam = 'head'
    if ref_cam in video_frames:
        num_frames = len(video_frames[ref_cam])
        if num_frames != len(states):
            raise ValueError(
                f"Frame count ({num_frames}) does not match state count ({len(states)}) "
                f"for episode {hdf5_path.stem}"
            )
        if (caption_list is not None and num_frames != len(caption_list)):
            print(
                f"Warning: Frame count ({num_frames}) does not match caption count ({len(caption_list)}) "
                f"for episode {hdf5_path.stem}. Ignoring captions."
            )
            caption_list = None
    
    return video_frames, depth_frames, states, actions, caption_list


def collect_data_files(source_dirs: List[Path], recursive: bool = True) -> List[Tuple[Path, dict, Path]]:
    """
    Collect all HDF5 and corresponding video files from source directories.
    Expects 3 videos per HDF5: _head.mp4, _left.mp4, _right.mp4

    Args:
        source_dirs: List of source directories
        recursive: If True, search recursively through nested directories

    Returns:
        List of (hdf5_path, video_paths_dict, caption_path) tuples
    """
    data_files = []

    for source_dir in source_dirs:
        # Use rglob for recursive search, glob for non-recursive
        if recursive:
            hdf5_files = []
            for root, _, files in os.walk(source_dir, followlinks=True):
                for name in files:
                    if name.endswith(".hdf5"):
                        hdf5_files.append(Path(root) / name)
            hdf5_files = sorted(hdf5_files)
        else:
            hdf5_files = sorted(source_dir.glob('*.hdf5'))

        for hdf5_file in hdf5_files:
            # Find corresponding video files
            # Expected format: {stem}_head.mp4, {stem}_left.mp4, {stem}_right.mp4
            # Note: stem might optionally include timestamp, but we assume
            # the HDF5 stem is the prefix for the videos.
            
            # If the HDF5 is "episode_0000.hdf5", stem is "episode_0000"
            # Videos should be "episode_0000_head.mp4", etc.
            
            # Only handle the specific format requested:
            # hdf5_path.stem() + '_head.mp4' / '_left.mp4' / '_right.mp4'
            
            stem = hdf5_file.stem
            parent = hdf5_file.parent
            
            video_head = parent / f"{stem}_head.mp4"
            video_left = parent / f"{stem}_left.mp4"
            video_right = parent / f"{stem}_right.mp4"
            
            videos = {
                "head": video_head,
                "left": video_left,
                "right": video_right
            }
            
            missing_videos = [k for k, v in videos.items() if not v.exists()]
            
            caption_file = hdf5_file.with_suffix('.json')
            caption_path = caption_file if caption_file.exists() else None

            if not missing_videos:
                data_files.append((hdf5_file, videos, caption_path))
            else:
                print(f"Warning: Missing videos for {hdf5_file}: {missing_videos}")

    return data_files


def main(
    source_dirs: List[str],
    repo_id: str = "cone_e_folding_clothes",
    output_dir: str = "fold_clothes_data_lerobot",
    robot_type: str = "dual_arm",
    fps: int = 10,
    downsample_factor: int = 4,
    test_split: float = 0.05,
    task_description: str = "Fold the shirt on the table",
    seed: int = 42,
    force_override: bool = False,
    push_to_hub: bool = False,
    recursive: bool = True,
):
    """
    Convert cone_e folding clothes data directly from HDF5+MP4 to LeRobot format.
    Creates separate train and validation datasets.

    Args:
        source_dirs: List of source directories containing HDF5 and video files
        repo_id: Repository ID for the output dataset
        output_dir: Local output directory for the dataset
        robot_type: Type of robot used for data collection
        fps: Frames per second after downsampling
        downsample_factor: Factor to downsample videos and actions
        test_split: Fraction of data to use for test set
        task_description: Natural language description of the task
        seed: Random seed for reproducibility
        force_override: Whether to override existing dataset
        push_to_hub: Whether to push the dataset to Hugging Face Hub
        recursive: Whether to search recursively through nested directories
    """
    # Convert paths
    source_paths = [Path(d) for d in source_dirs]
    output_path = Path(output_dir)

    # Create separate output directories for train and validation
    train_output_path = output_path / "train"
    val_output_path = output_path / "validation"

    # Clean up any existing dataset if force_override
    if output_path.exists():
        if force_override:
            print(f"Removing existing dataset at {output_path}")
            shutil.rmtree(output_path)
        else:
            raise ValueError(
                f"Dataset already exists at {output_path}. "
                "Use --force_override to overwrite."
            )

    # Set random seed for reproducibility
    rng = np.random.default_rng(seed)

    print(f"Collecting data files from {len(source_paths)} directories...")
    data_files = collect_data_files(source_paths, recursive=recursive)
    print(f"Found {len(data_files)} episodes")

    if len(data_files) == 0:
        raise ValueError("No data files found!")

    # Shuffle and split data
    indices = np.arange(len(data_files))
    rng.shuffle(indices)

    n_test = int(len(data_files) * test_split)
    n_train = len(data_files) - n_test

    test_indices = set(indices[:n_test])
    train_indices = set(indices[n_test:])

    print(f"Train episodes: {n_train}")
    print(f"Test episodes: {n_test}")
    print(f"Downsample factor: {downsample_factor}")
    print(f"Output FPS: {fps}")

    # Load first episode to determine dimensions
    print("\nLoading first episode to determine dimensions...")
    first_hdf5, first_videos, first_caption = data_files[0]
    sample_frames, sample_depth_frames, sample_states, sample_actions, _ = load_episode_from_hdf5(
        first_hdf5, first_videos, first_caption, downsample_factor
    )

    frame_shape = sample_frames['head'][0].shape  # (H, W, 3)
    depth_shape = sample_depth_frames[0].shape  # (H, W)
    state_dim = sample_states.shape[1]  # Should be 20
    action_dim = sample_actions.shape[1]  # Should be 20

    print(f"Frame shape: {frame_shape}")
    print(f"Depth shape: {depth_shape}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    # Define feature specification (shared between both datasets)
    features = {
        "observation.image.head": {
            "dtype": "video",
            "shape": frame_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.image.left_wrist": {
            "dtype": "video",
            "shape": frame_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.image.right_wrist": {
            "dtype": "video",
            "shape": frame_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.depth": {
            "dtype": "float32",
            "shape": depth_shape,
            "names": ["height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [
                "left_x", "left_y", "left_z",
                "left_r11", "left_r21", "left_r31",
                "left_r12", "left_r22", "left_r32",
                "left_gripper",
                "right_x", "right_y", "right_z",
                "right_r11", "right_r21", "right_r31",
                "right_r12", "right_r22", "right_r32",
                "right_gripper",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [
                "left_dx", "left_dy", "left_dz",
                "left_dr11", "left_dr21", "left_dr31",
                "left_dr12", "left_dr22", "left_dr32",
                "left_gripper",
                "right_dx", "right_dy", "right_dz",
                "right_dr11", "right_dr21", "right_dr31",
                "right_dr12", "right_dr22", "right_dr32",
                "right_gripper",
            ],
        },
    }

    # Create TWO separate LeRobot datasets - one for train, one for validation
    print(f"\nCreating train dataset: {repo_id}_train")
    train_dataset = LeRobotDataset.create(
        repo_id=f"{repo_id}_train",
        root=train_output_path,
        robot_type=robot_type,
        fps=fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    print(f"Creating validation dataset: {repo_id}_validation")
    val_dataset = LeRobotDataset.create(
        repo_id=f"{repo_id}_validation",
        root=val_output_path,
        robot_type=robot_type,
        fps=fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Process each episode
    print("\nProcessing episodes...")
    train_count = 0
    test_count = 0

    for idx, (hdf5_path, video_paths, caption_path) in enumerate(data_files):
        try:
            # Load episode data
            video_frames, depth_frames, states, actions, captions = load_episode_from_hdf5(
                hdf5_path, video_paths, caption_path, downsample_factor
            )

            # Determine split and select appropriate dataset
            split_name = "test" if idx in test_indices else "train"
            current_dataset = val_dataset if split_name == "test" else train_dataset

            # Add frames to the dataset
            num_frames = len(states)
            for i in range(num_frames):
                frame_dict = {
                    "observation.image.head": video_frames['head'][i],
                    "observation.image.left_wrist": video_frames['left'][i],
                    "observation.image.right_wrist": video_frames['right'][i],
                    "observation.depth": depth_frames[i].astype(np.float32),
                    "observation.state": states[i].astype(np.float32),
                    "action": actions[i].astype(np.float32),
                    "task": task_description,
                }
                current_dataset.add_frame(frame_dict)

            # Mark end of episode on the appropriate dataset
            current_dataset.save_episode()

            if split_name == "train":
                train_count += 1
            else:
                test_count += 1

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(data_files)} episodes "
                      f"(train: {train_count}, test: {test_count})")

        except Exception as e:
            print(f"Error processing {hdf5_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\nDataset conversion complete!")
    print(f"\nTrain dataset:")
    print(f"  Location: {train_output_path}")
    print(f"  Episodes: {train_dataset.num_episodes}")
    print(f"  Frames: {train_dataset.num_frames}")

    print(f"\nValidation dataset:")
    print(f"  Location: {val_output_path}")
    print(f"  Episodes: {val_dataset.num_episodes}")
    print(f"  Frames: {val_dataset.num_frames}")

    # Push to hub if requested
    if push_to_hub:
        print(f"\nPushing train dataset to Hugging Face Hub: {repo_id}_train")
        train_dataset.push_to_hub(repo_id=f"{repo_id}_train")

        print(f"Pushing validation dataset to Hugging Face Hub: {repo_id}_validation")
        val_dataset.push_to_hub(repo_id=f"{repo_id}_validation")

        print("Datasets successfully pushed to hub!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert cone_e folding clothes data directly to LeRobot format with separate train/validation datasets"
    )
    parser.add_argument(
        "--source_dirs",
        nargs='+',
        default=['shirt_folding_simple_cone_e',
                 'shirt_folding_simple_cone_e_2',
                 'shirt_folding_simple_cone_e_3'],
        help="Source directories containing HDF5 and video files"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="cone_e_folding_clothes",
        help="Repository ID base for the output datasets (e.g., 'username/dataset_name'). "
             "Will create 'repo_id_train' and 'repo_id_validation'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="fold_clothes_data_lerobot",
        help="Local output directory for the dataset (will create train/ and validation/ subdirs)"
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        default="dual_arm",
        help="Type of robot used for data collection",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second after downsampling (original_fps / downsample_factor)",
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=4,
        help="Factor to downsample videos and actions (1 = no downsampling)"
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.05,
        help="Fraction of data to use for test set (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default="Fold the shirt on the table",
        help="Natural language description of the task",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--force_override",
        action="store_true",
        help="Override existing dataset if it exists",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the dataset to Hugging Face Hub after creation",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search recursively through nested directories (default: True)",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Do not search recursively (only search top-level directory)",
    )

    args = parser.parse_args()

    main(
        source_dirs=args.source_dirs,
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        robot_type=args.robot_type,
        fps=args.fps,
        downsample_factor=args.downsample_factor,
        test_split=args.test_split,
        task_description=args.task_description,
        seed=args.seed,
        force_override=args.force_override,
        push_to_hub=args.push_to_hub,
        recursive=args.recursive,
    )