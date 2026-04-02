"""Camera feed management and display."""

import cv2
import time
import threading
import numpy as np
from record3d import Record3DStream


class CameraFeedManager:
    """Manages iPhone camera feed and live display."""

    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event
        self.latest_rgb_frame = None
        self.latest_depth_frame = None
        self.latest_rgb_timestamp = None
        self.rgb_frame_lock = threading.Lock()
        self.session = None
        self.frame_event = threading.Event()

        self.is_episode_active = False
        self.episode_start_time = None
        self.is_recording = False
        self.autonomous_mode = False

        # Optional wrist camera reference (set externally)
        self.wrist_camera = None

    def _on_new_frame(self):
        self.frame_event.set()

    def start(self):
        self.iphone_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.iphone_thread.start()
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()

    def stop(self):
        self.iphone_thread.join(timeout=2.0)
        self.display_thread.join(timeout=2.0)

    def get_latest_frame(self):
        with self.rgb_frame_lock:
            if self.latest_rgb_frame is not None:
                if self.latest_depth_frame is not None:
                    return self.latest_rgb_frame.copy(), self.latest_rgb_timestamp, self.latest_depth_frame.copy()
                else:
                    return self.latest_rgb_frame.copy(), self.latest_rgb_timestamp, None
            return None, None, None

    def _capture_loop(self):
        print('Searching for iPhone devices')
        devs = Record3DStream.get_connected_devices()
        print(f'{len(devs)} device(s) found')

        if len(devs) == 0:
            print("WARNING: No iPhone devices found. RGB recording disabled.")
            return

        dev = devs[0]
        self.session = Record3DStream()
        self.session.on_new_frame = self._on_new_frame
        self.session.on_stream_stopped = lambda: print('iPhone stream stopped')
        self.session.connect(dev)

        print("iPhone connected (head camera), starting RGB capture")

        while not self.stop_event.is_set():
            self.frame_event.wait(timeout=0.1)
            if self.session is None:
                continue
            try:
                rgb = np.array(self.session.get_rgb_frame())
                depth = np.array(self.session.get_depth_frame())
                frame_receive_time = time.time()
                with self.rgb_frame_lock:
                    self.latest_rgb_frame = rgb
                    self.latest_depth_frame = depth
                    self.latest_rgb_timestamp = frame_receive_time
            except Exception as e:
                print(f"Error getting RGB frame: {e}")
            self.frame_event.clear()

    def _display_loop(self):
        """Single display thread that shows head camera and optional wrist camera."""
        head_window = "Head Camera"
        wrist_window = "Right Wrist Camera"

        cv2.namedWindow(head_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(head_window, 640, 480)

        wrist_window_created = False

        print("Camera feed display started")

        while not self.stop_event.is_set():
            # --- Head camera ---
            with self.rgb_frame_lock:
                if self.latest_rgb_frame is not None:
                    display_frame = cv2.rotate(self.latest_rgb_frame.copy(), cv2.ROTATE_90_CLOCKWISE)
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                    status_text = self._build_status_text()
                    y_offset = 30
                    for text in status_text:
                        cv2.putText(display_frame, text, (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        y_offset += 30
                    cv2.imshow(head_window, display_frame)
                else:
                    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(black_frame, "Waiting for head camera...", (120, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imshow(head_window, black_frame)

            # --- Wrist camera ---
            if self.wrist_camera is not None:
                if not wrist_window_created:
                    cv2.namedWindow(wrist_window, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(wrist_window, 640, 480)
                    wrist_window_created = True

                wrist_frame, _, _ = self.wrist_camera.get_latest_frame()
                if wrist_frame is not None:
                    wrist_display = cv2.rotate(wrist_frame, cv2.ROTATE_90_CLOCKWISE)
                    wrist_display = cv2.cvtColor(wrist_display, cv2.COLOR_RGB2BGR)
                    cv2.imshow(wrist_window, wrist_display)
                else:
                    black = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(black, "Waiting for wrist camera...", (100, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imshow(wrist_window, black)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop_event.set()
                break

            time.sleep(0.033)

        cv2.destroyWindow(head_window)
        if wrist_window_created:
            cv2.destroyWindow(wrist_window)
        print("Camera feed display stopped")

    def _build_status_text(self):
        status_text = []
        if self.is_episode_active:
            status_text.append("EPISODE ACTIVE")
            if self.episode_start_time is not None:
                elapsed = time.time() - self.episode_start_time
                status_text.append(f"Time: {elapsed:.1f}s")
        else:
            status_text.append("EPISODE INACTIVE")
        if self.is_recording:
            status_text.append("RECORDING")
        if self.autonomous_mode:
            status_text.append("AUTO MODE")
        return status_text


class USBWristCameraFeedManager:
    """Manages a single USB wrist camera feed via Record3D (capture only, no display)."""

    def __init__(self, stop_event, device_index=1, label="right wrist"):
        self.stop_event = stop_event
        self.device_index = device_index
        self.label = label
        self.session = None
        self.frame_event = threading.Event()
        self.lock = threading.Lock()
        self.latest_rgb = None
        self.latest_ts = None

    def start(self):
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.thread.join(timeout=2.0)

    def get_latest_frame(self):
        with self.lock:
            if self.latest_rgb is not None and self.latest_rgb.size > 0:
                return self.latest_rgb.copy(), self.latest_ts, None
            return None, None, None

    def _capture_loop(self):
        devs = Record3DStream.get_connected_devices()
        if self.device_index >= len(devs):
            print(f"[{self.label}] Device index {self.device_index} not found ({len(devs)} devices)")
            return

        self.session = Record3DStream()
        self.session.on_new_frame = lambda: self.frame_event.set()
        self.session.on_stream_stopped = lambda: print(f"[{self.label}] stream stopped")
        self.session.connect(devs[self.device_index])
        print(f"[{self.label}] Connected to device {self.device_index}")

        while not self.stop_event.is_set():
            self.frame_event.wait(timeout=0.1)
            if self.session is None:
                continue
            try:
                rgb = np.array(self.session.get_rgb_frame())
                ts = time.time()
                with self.lock:
                    self.latest_rgb = rgb
                    self.latest_ts = ts
            except Exception:
                pass
            self.frame_event.clear()