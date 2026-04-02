"""Data recording functionality for robot episodes."""

import cv2
import h5py
import time
import queue
import imageio
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import namedtuple


# Data structure for recording queue
RecordingSample = namedtuple('RecordingSample', [
    'timestamp',
    'left_ee_pose',
    'right_ee_pose',
    'left_gripper_exact',
    'right_gripper_exact',
    'left_gripper',
    'right_gripper',
    'rgb_frame',
    'depth_frame',
    'rgb_timestamp',
    'left_joint_positions',
    'right_joint_positions',
    'left_wrist_rgb_frame',
    'right_wrist_rgb_frame',
])


class DataRecorder:
    """Handles recording and saving of robot episode data."""

    def __init__(self, save_dir: Path, stop_event: threading.Event):
        """
        Initialize data recorder.

        Args:
            save_dir: Directory to save recordings
            stop_event: Event to signal shutdown
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.stop_event = stop_event

        self.is_recording = False
        self.episode_count = 0
        self.episode_data = self._init_episode_data()

        # Recording queue and worker thread
        self.recording_queue = queue.Queue(maxsize=100)
        self.recording_thread = threading.Thread(target=self._recording_worker, daemon=True)
        self.recording_thread.start()

        self.episode_lock = threading.Lock()

        print(f"Recording enabled. Data will be saved to: {self.save_dir}")

    def _init_episode_data(self):
        """Initialize empty episode data structure."""
        return {
            'timestamps': [],
            'left_ee_pos': [],
            'left_ee_quat': [],
            'left_gripper_exact': [],
            'left_gripper': [],
            'right_ee_pos': [],
            'right_ee_quat': [],
            'right_gripper_exact': [],
            'right_gripper': [],
            'rgb_frames': [],
            'left_wrist_rgb_frames': [],
            'right_wrist_rgb_frames': [],
            'depth_frames': [],
            'rgb_frame_timestamps': [],
            'left_joint_positions': [],
            'right_joint_positions': [],
        }

    def start_episode(self):
        """Start recording a new episode."""
        self.is_recording = True
        self.episode_data = self._init_episode_data()
        print(f"\n🔴 RECORDING STARTED - Episode {self.episode_count}")

    def end_episode(self):
        """Stop recording and save episode data."""
        if not self.is_recording:
            return

        self.is_recording = False

        # Wait for queue to be processed
        while not self.recording_queue.empty():
            time.sleep(0.01)

        # Save episode
        num_samples = len(self.episode_data['timestamps'])
        if num_samples > 0:
            self._save_episode()
            print(f"⚫ RECORDING STOPPED - Saved {num_samples} samples")
        else:
            print("⚫ RECORDING STOPPED - No samples recorded")

        self.episode_count += 1

    def record_sample(self, sample: RecordingSample):
        """
        Add a sample to the recording queue.

        Args:
            sample: RecordingSample namedtuple with observation data
        """
        if not self.is_recording:
            return

        try:
            self.recording_queue.put_nowait(sample)
        except queue.Full:
            print("WARNING: Recording queue full, dropping sample!")

    def stop(self):
        """Stop recorder thread."""
        self.recording_thread.join(timeout=2.0)

    def _recording_worker(self):
        """Background thread that processes recorded samples."""
        while not self.stop_event.is_set():
            try:
                sample = self.recording_queue.get(timeout=0.1)

                # Process the sample: rotate frames to landscape orientation
                rgb_frame = sample.rgb_frame
                depth_frame = sample.depth_frame
                left_wrist_rgb_frame = sample.left_wrist_rgb_frame
                right_wrist_rgb_frame = sample.right_wrist_rgb_frame

                if rgb_frame is not None:
                    rgb_frame = cv2.rotate(rgb_frame, cv2.ROTATE_90_CLOCKWISE)
                if depth_frame is not None:
                    depth_frame = cv2.rotate(depth_frame, cv2.ROTATE_90_CLOCKWISE)
                if left_wrist_rgb_frame is not None:
                    left_wrist_rgb_frame = cv2.rotate(left_wrist_rgb_frame, cv2.ROTATE_90_CLOCKWISE)
                if right_wrist_rgb_frame is not None:
                    right_wrist_rgb_frame = cv2.rotate(right_wrist_rgb_frame, cv2.ROTATE_90_CLOCKWISE)

                # Extract position and quaternion from left & right ee pose
                left_pos = sample.left_ee_pose.translation()
                left_quat = sample.left_ee_pose.rotation().wxyz
                right_pos = sample.right_ee_pose.translation()
                right_quat = sample.right_ee_pose.rotation().wxyz

                with self.episode_lock:
                    self.episode_data['left_ee_pos'].append(left_pos)
                    self.episode_data['left_ee_quat'].append(left_quat)
                    self.episode_data['left_gripper_exact'].append(sample.left_gripper_exact)
                    self.episode_data['left_gripper'].append(sample.left_gripper)
                    self.episode_data['left_joint_positions'].append(sample.left_joint_positions)
                    self.episode_data['timestamps'].append(sample.timestamp)
                    self.episode_data['right_ee_pos'].append(right_pos)
                    self.episode_data['right_ee_quat'].append(right_quat)
                    self.episode_data['right_gripper_exact'].append(sample.right_gripper_exact)
                    self.episode_data['right_gripper'].append(sample.right_gripper)
                    self.episode_data['right_joint_positions'].append(sample.right_joint_positions)

                    # Store head RGB frame
                    self.episode_data['rgb_frames'].append(
                        rgb_frame if rgb_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                    )
                    self.episode_data['rgb_frame_timestamps'].append(
                        sample.rgb_timestamp if sample.rgb_timestamp is not None else sample.timestamp
                    )

                    # Store wrist RGB frames
                    self.episode_data['left_wrist_rgb_frames'].append(
                        left_wrist_rgb_frame if left_wrist_rgb_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                    )
                    self.episode_data['right_wrist_rgb_frames'].append(
                        right_wrist_rgb_frame if right_wrist_rgb_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                    )

                    # Store depth frame
                    self.episode_data['depth_frames'].append(
                        depth_frame if depth_frame is not None else np.zeros((480, 640), dtype=np.float32)
                    )

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in recording worker: {e}")

    def _save_episode(self):
        """Save episode data to HDF5 and video files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_name = f"episode_{self.episode_count:04d}_{timestamp}"

        # Save to HDF5
        h5_path = self.save_dir / f"{episode_name}.hdf5"
        print(f"Saving episode to {h5_path}")

        try:
            with h5py.File(h5_path, 'w') as f:
                with self.episode_lock:
                    # assert data lengths are consistent
                    num_samples = len(self.episode_data['timestamps'])
                    for key, data_list in self.episode_data.items():
                        assert len(data_list) == num_samples, f"Data length mismatch for {key}"

                    # Save timestamps
                    f.create_dataset('timestamps', data=np.array(self.episode_data['timestamps']))

                    # Save left arm data
                    f.create_dataset('left_ee_pos', data=np.array(self.episode_data['left_ee_pos']))
                    f.create_dataset('left_ee_quat', data=np.array(self.episode_data['left_ee_quat']))
                    f.create_dataset('left_gripper_exact', data=np.array(self.episode_data['left_gripper_exact']))
                    f.create_dataset('left_gripper', data=np.array(self.episode_data['left_gripper']))
                    f.create_dataset('left_joint_positions', data=np.array(self.episode_data['left_joint_positions']))

                    # Save right arm data
                    f.create_dataset('right_ee_pos', data=np.array(self.episode_data['right_ee_pos']))
                    f.create_dataset('right_ee_quat', data=np.array(self.episode_data['right_ee_quat']))
                    f.create_dataset('right_gripper_exact', data=np.array(self.episode_data['right_gripper_exact']))
                    f.create_dataset('right_gripper', data=np.array(self.episode_data['right_gripper']))
                    f.create_dataset('right_joint_positions', data=np.array(self.episode_data['right_joint_positions']))

                    # Save RGB frames
                    # rgb_frames = np.array(self.episode_data['rgb_frames'])
                    # f.create_dataset('rgb_frames', data=rgb_frames, compression='gzip')
                    f.create_dataset('rgb_frame_timestamps', data=np.array(self.episode_data['rgb_frame_timestamps']))
                    # Save depth frames
                    depth_frames = np.array(self.episode_data['depth_frames'])
                    f.create_dataset('depth_frames', data=depth_frames, compression='gzip')

                    # Save metadata
                    f.attrs['num_samples'] = len(self.episode_data['timestamps'])
                f.attrs['episode_number'] = self.episode_count
                f.attrs['timestamp'] = timestamp

            print(f"  ✓ Saved HDF5 with {len(self.episode_data['timestamps'])} samples")
            self.episode_data['rgb_frame_timestamps']
        except Exception as e:
            print(f"ERROR saving HDF5: {e}")

        # Save video
        self._save_video(episode_name)

    def _save_video(self, episode_name: str):
        """Save head, left-wrist, and right-wrist RGB frames as separate MP4s."""
        videos = [
            (f"{episode_name}_head.mp4",  self.episode_data['rgb_frames']),
            (f"{episode_name}_left.mp4",  self.episode_data['left_wrist_rgb_frames']),
            (f"{episode_name}_right.mp4", self.episode_data['right_wrist_rgb_frames']),
        ]

        for filename, frames in videos:
            if not frames:
                continue
            video_path = self.save_dir / filename
            try:
                writer = imageio.get_writer(str(video_path), fps=30, codec='libx264', quality=8)
                for frame in frames:
                    writer.append_data(frame)
                writer.close()
                height, width = frames[0].shape[:2]
                print(f"  ✓ Saved video: {video_path} ({len(frames)} frames, {width}x{height})")
            except Exception as e:
                print(f"ERROR saving video {filename}: {e}")