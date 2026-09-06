"""
仕様：
10秒に一度VLAを呼び出し、写真撮影→終了条件の判定
出力:
1.実行中
2.成功
3.時間切れ（1分）
4.失敗（自律的に修復不可）

使用したVLA: Gemini Robotics-ER 1.6


"""
import cv2
import time
import numpy as np
from google import genai
from google.genai import types
import argparse
from enum import Enum
import threading
from camera import CameraFeedManager
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


MODEL_NAME = "gemini-robotics-er-1.6-preview"
POLL_INTERVAL_SEC = 10
TIMEOUT_SEC = 60
CAMERA_DEVICE_HEAD = 0   

class SubtaskStatus(Enum):
    RUNNING = 1   # 実行中
    SUCCESS = 2   # 成功
    TIMEOUT = 3   # 時間切れ
    FAILURE = 4   # 失敗（修復不可）


class VLASubtaskRunner:
    """10秒ごとにカメラ撮影→Geminiで終了条件を判定する"""

    TIMEOUT_SEC = 60.0
    INTERVAL_SEC = 10.0

    def __init__(self, camera_manager, task_description: str, api_key: str):
        self.camera_manager = camera_manager
        self.task_description = task_description
        self.client = genai.Client(api_key=api_key)
        self.status = SubtaskStatus.RUNNING

    def run(self) -> SubtaskStatus:
        """サブタスクを実行し、最終ステータスを返す。"""
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time

            # タイムアウト確認
            if elapsed >= self.TIMEOUT_SEC:
                print("タイムアウト（1分経過）")
                self.status = SubtaskStatus.TIMEOUT
                return self.status

            # geminiで判定
            self.status = self._call_gemini_with_cameras()
            print(f"ステータス: {self.status.name} （経過: {elapsed:.1f}s）")

            if self.status != SubtaskStatus.RUNNING:
                return self.status

            time.sleep(self.INTERVAL_SEC)

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        """RGBフレームをJPEGバイト列に変換。"""
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", bgr)
        return buf.tobytes()

    def _call_gemini_with_cameras(self) -> SubtaskStatus:
        """全カメラのフレームを取得し判定する。"""
        prompt = f"""
You are an AI monitoring a robot subtask.
Look at the camera images below and determine the current status of the subtask.

Subtask: {self.task_description}

Reply with exactly one of the following words (no other text):
- RUNNING   : The task is still in progress
- SUCCESS   : The task has been completed successfully
- FAILURE   : The task has failed and cannot be recovered autonomously
"""
        parts = [types.Part.from_text(text=prompt)]

        # ヘッドカメラ
        frame, _, _ = self.camera_manager.get_latest_frame()
        if frame is not None:
            parts.append(types.Part.from_bytes(
                data=self._encode_frame(frame), mime_type="image/jpeg"
            ))

        # 右手首カメラ
        if self.camera_manager.wrist_camera is not None:
            frame, _, _ = self.camera_manager.wrist_camera.get_latest_frame()
            if frame is not None:
                parts.append(types.Part.from_bytes(
                    data=self._encode_frame(frame), mime_type="image/jpeg"
                ))

        # 左手首カメラ
        if self.camera_manager.left_wrist_camera is not None:
            frame, _, _ = self.camera_manager.left_wrist_camera.get_latest_frame()
            if frame is not None:
                parts.append(types.Part.from_bytes(
                    data=self._encode_frame(frame), mime_type="image/jpeg"
                ))

        # 画像が1枚もなければRUNNINGを返す
        if len(parts) == 1:
            print("failed to get frame")
            return SubtaskStatus.RUNNING

        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Content(role="user", parts=parts)],
        )

        answer = response.text.strip().upper()
        print(f"[Gemini] answer: {answer}")

        if "SUCCESS" in answer:
            return SubtaskStatus.SUCCESS
        elif "FAILURE" in answer:
            return SubtaskStatus.FAILURE
        else:
            return SubtaskStatus.RUNNING


if __name__ == "__main__":
    import argparse
    from camera import CameraFeedManager

    parser = argparse.ArgumentParser(description="VLA subtask runner")
    parser.add_argument("--task", type=str, required=True, help="サブタスクの説明")
    parser.add_argument("--interval", type=float, default=10.0, help="撮影間隔（秒）")
    parser.add_argument("--timeout", type=float, default=60.0, help="タイムアウト（秒）")
    args = parser.parse_args()


    stop_event = threading.Event()
    camera = CameraFeedManager(stop_event)
    camera.start()


    runner = VLASubtaskRunner(
        camera_manager=camera,
        task_description=args.task,
        api_key=os.environ["GOOGLE_API_KEY"],
    )
    runner.INTERVAL_SEC = args.interval
    runner.TIMEOUT_SEC = args.timeout

    print(f"task: {args.task}")
    print(f"interval: {args.interval}s / timeout: {args.timeout}s")
    print("=" * 40)

    result = runner.run()
    print(f"\nresult: {result.name}")

    stop_event.set()
    camera.stop()