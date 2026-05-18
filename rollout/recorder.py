"""Data recording functionality for robot episodes."""

import cv2
import h5py
import time
import queue
import subprocess
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


class StreamingVideoWriter:
    """Crash-safe video writer using ffmpeg fragmented MP4.
    
    Uses -movflags frag_keyframe+empty_moov so the file is always
    playable even if the process is killed without cleanup.
    Writer is created lazily on first frame using actual dimensions.
    """

    def __init__(self, path, fps=30):
        self.path = str(path)
        self.fps = fps
        self._proc = None
        self._frame_count = 0
        self._width = None
        self._height = None

    def _start_ffmpeg(self, width, height):
        """Start ffmpeg subprocess for writing fragmented MP4."""
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgb24',
            '-r', str(self.fps),
            '-i', '-',
            '-an',
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-crf', '23',
            '-movflags', 'frag_keyframe+empty_moov',
            self.path,
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._width = width
        self._height = height
        print(f"  📹 Video writer opened: {self.path} ({width}x{height})")

    def write(self, frame_rgb):
        """Write an RGB frame. Creates ffmpeg process on first call."""
        if frame_rgb is None:
            return
        h, w = frame_rgb.shape[:2]
        if self._proc is None:
            self._start_ffmpeg(w, h)
        try:
            self._proc.stdin.write(frame_rgb.astype(np.uint8).tobytes())
            self._frame_count += 1
        except (BrokenPipeError, OSError):
            pass

    def release(self):
        """Close ffmpeg process gracefully."""
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            self._proc = None

    @property
    def frame_count(self):
        return self._frame_count


class DataRecorder:
    """Handles recording and saving of robot episode data.

    Video frames are streamed directly to MP4 files via ffmpeg
    instead of accumulating in memory, preventing OOM on long episodes.
    Fragmented MP4 ensures files are playable even after crashes.
    """

    def __init__(self, save_dir: Path, stop_event: threading.Event):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.stop_event = stop_event

        self.is_recording = False
        self.episode_count = 0
        self.episode_data = self._init_episode_data()

        # Video writers
        self._head_writer = None
        self._left_writer = None
        self._right_writer = None
        self._episode_name = None

        # Recording queue and worker thread
        self.recording_queue = queue.Queue(maxsize=100)
        self.recording_thread = threading.Thread(target=self._recording_worker, daemon=True)
        self.recording_thread.start()

        self.episode_lock = threading.Lock()

        print(f"Recording enabled. Data will be saved to: {self.save_dir}")

    def _init_episode_data(self):
        """Initialize empty episode data structure (no frame lists)."""
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
            'rgb_frame_timestamps': [],
            'left_joint_positions': [],
            'right_joint_positions': [],
        }

    def start_episode(self):
        """Start recording a new episode. Video writers created lazily on first frame."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._episode_name = f"episode_{self.episode_count:04d}_{timestamp}"

        # Create lazy video writers (ffmpeg started on first frame)
        self._head_writer = StreamingVideoWriter(
            self.save_dir / f"{self._episode_name}_head.mp4"
        )
        self._left_writer = StreamingVideoWriter(
            self.save_dir / f"{self._episode_name}_left.mp4"
        )
        self._right_writer = StreamingVideoWriter(
            self.save_dir / f"{self._episode_name}_right.mp4"
        )

        self.episode_data = self._init_episode_data()
        self.is_recording = True
        print(f"\n🔴 RECORDING STARTED - Episode {self.episode_count}")

    def end_episode(self):
        """Stop recording, close video writers, save HDF5."""
        if not self.is_recording:
            return

        self.is_recording = False

        # Wait for queue to drain
        while not self.recording_queue.empty():
            time.sleep(0.01)

        # Close video writers
        for name, writer in [("head", self._head_writer),
                             ("left", self._left_writer),
                             ("right", self._right_writer)]:
            if writer is not None:
                print(f"  ✓ Closed {name} video ({writer.frame_count} frames)")
                writer.release()
        self._head_writer = None
        self._left_writer = None
        self._right_writer = None

        # Save HDF5 (scalar data only, very fast)
        num_samples = len(self.episode_data['timestamps'])
        if num_samples > 0:
            self._save_hdf5()
            print(f"⚫ RECORDING STOPPED - Saved {num_samples} samples")
        else:
            print("⚫ RECORDING STOPPED - No samples recorded")

        self.episode_count += 1

    def record_sample(self, sample: RecordingSample):
        """Add a sample to the recording queue."""
        if not self.is_recording:
            return
        try:
            self.recording_queue.put_nowait(sample)
        except queue.Full:
            print("WARNING: Recording queue full, dropping sample!")

    def stop(self):
        """Stop recorder thread."""
        if self.is_recording:
            self.end_episode()
        self.recording_thread.join(timeout=2.0)

    def _recording_worker(self):
        """Background thread that processes recorded samples."""
        while not self.stop_event.is_set():
            try:
                sample = self.recording_queue.get(timeout=0.1)

                # Rotate raw frames to landscape orientation
                rgb_frame = sample.rgb_frame
                left_wrist_rgb_frame = sample.left_wrist_rgb_frame
                right_wrist_rgb_frame = sample.right_wrist_rgb_frame

                if rgb_frame is not None:
                    rgb_frame = cv2.rotate(rgb_frame, cv2.ROTATE_90_CLOCKWISE)
                if left_wrist_rgb_frame is not None:
                    left_wrist_rgb_frame = cv2.rotate(left_wrist_rgb_frame, cv2.ROTATE_90_CLOCKWISE)
                if right_wrist_rgb_frame is not None:
                    right_wrist_rgb_frame = cv2.rotate(right_wrist_rgb_frame, cv2.ROTATE_90_CLOCKWISE)

                # Stream frames to disk immediately (no memory accumulation)
                self._head_writer.write(rgb_frame)
                self._left_writer.write(left_wrist_rgb_frame)
                self._right_writer.write(right_wrist_rgb_frame)

                # Extract pose data
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
                    self.episode_data['rgb_frame_timestamps'].append(
                        sample.rgb_timestamp if sample.rgb_timestamp is not None else sample.timestamp
                    )

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in recording worker: {e}")

    def _save_hdf5(self):
        """Save scalar episode data to HDF5 (no frames — those are in MP4s)."""
        h5_path = self.save_dir / f"{self._episode_name}.hdf5"
        print(f"Saving episode to {h5_path}")

        try:
            with h5py.File(h5_path, 'w') as f:
                with self.episode_lock:
                    f.create_dataset('timestamps', data=np.array(self.episode_data['timestamps']))
                    f.create_dataset('left_ee_pos', data=np.array(self.episode_data['left_ee_pos']))
                    f.create_dataset('left_ee_quat', data=np.array(self.episode_data['left_ee_quat']))
                    f.create_dataset('left_gripper_exact', data=np.array(self.episode_data['left_gripper_exact']))
                    f.create_dataset('left_gripper', data=np.array(self.episode_data['left_gripper']))
                    f.create_dataset('left_joint_positions', data=np.array(self.episode_data['left_joint_positions']))
                    f.create_dataset('right_ee_pos', data=np.array(self.episode_data['right_ee_pos']))
                    f.create_dataset('right_ee_quat', data=np.array(self.episode_data['right_ee_quat']))
                    f.create_dataset('right_gripper_exact', data=np.array(self.episode_data['right_gripper_exact']))
                    f.create_dataset('right_gripper', data=np.array(self.episode_data['right_gripper']))
                    f.create_dataset('right_joint_positions', data=np.array(self.episode_data['right_joint_positions']))
                    f.create_dataset('rgb_frame_timestamps', data=np.array(self.episode_data['rgb_frame_timestamps']))
                    f.attrs['num_samples'] = len(self.episode_data['timestamps'])
                f.attrs['episode_number'] = self.episode_count
                f.attrs['timestamp'] = self._episode_name.split('_', 2)[-1]

            print(f"  ✓ Saved HDF5 with {len(self.episode_data['timestamps'])} samples")
        except Exception as e:
            print(f"ERROR saving HDF5: {e}")