#!/usr/bin/env python3
"""Identify Record3D cameras by showing each view.
Run: python robot/camera_id.py
"""

import json, os, time, threading
from pathlib import Path
import numpy as np
import cv2
from record3d import Record3DStream

CAM_MAP_FILE = Path(__file__).parent / "camera_map.json"


def identify_cameras():
    devs = Record3DStream.get_connected_devices()
    n = len(devs)
    print(f"{n} Record3D device(s) found")
    if n == 0:
        return {}

    mapping = {}
    remaining = ["head", "right", "left"]

    for i in range(min(n, 3)):
        frame_holder = [None]
        event = threading.Event()

        s = Record3DStream()
        def on_frame(sess=s, fh=frame_holder, ev=event):
            try:
                fh[0] = np.array(sess.get_rgb_frame())
                ev.set()
            except:
                pass
        s.on_new_frame = on_frame
        s.on_stream_stopped = lambda: None

        try:
            s.connect(devs[i])
        except Exception as e:
            print(f"  Device {i}: failed ({e})")
            continue

        # Wait for frame
        event.wait(timeout=3.0)
        if frame_holder[0] is None:
            print(f"  Device {i}: no frame received")
            continue

        # Show view
        frame = cv2.rotate(frame_holder[0], cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(frame, f"Device {i} - Which camera?", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Remaining: {remaining}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Identify Camera", frame)
        cv2.waitKey(500)

        label = input(f"  Device {i} — which camera? ({'/'.join(remaining)}): ").strip().lower()
        cv2.destroyAllWindows()

        if label in remaining:
            mapping[label] = i
            remaining.remove(label)
        else:
            print(f"  '{label}' not recognized, skipping")

        if not remaining:
            break

    CAM_MAP_FILE.write_text(json.dumps(mapping, indent=2))
    print(f"\nSaved to {CAM_MAP_FILE}: {mapping}")
    return mapping


def load_camera_map():
    if CAM_MAP_FILE.exists():
        return json.loads(CAM_MAP_FILE.read_text())
    print(f"[WARN] No camera map. Run: python robot/camera_id.py")
    return {"head": 0, "right": 1, "left": 2}


def connected_camera_udids():
    """Return the live Record3D index->UDID mapping without opening streams."""

    devices = Record3DStream.get_connected_devices()
    return {
        index: str(getattr(device, "udid", ""))
        for index, device in enumerate(devices)
    }


def configure_camera_map_by_udid(expected_udids, *, preserve_unknown=True):
    """Atomically configure named cameras from immutable Record3D UDIDs.

    Enumeration order is not stable after reconnects.  Autonomous runs call
    this before opening a stream, so an index reorder cannot silently swap the
    head and wrist views.
    """

    live = connected_camera_udids()
    reverse = {udid: index for index, udid in live.items() if udid}
    missing = {
        name: udid
        for name, udid in expected_udids.items()
        if str(udid) not in reverse
    }
    if missing:
        raise RuntimeError(
            f"expected Record3D camera UDIDs are not connected: {missing}; "
            f"connected={live}"
        )
    mapping = load_camera_map() if preserve_unknown else {}
    mapping.update(
        {name: reverse[str(udid)] for name, udid in expected_udids.items()}
    )
    temporary = CAM_MAP_FILE.with_name(
        f".{CAM_MAP_FILE.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    )
    encoded = (json.dumps(mapping, indent=2) + "\n").encode()
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, CAM_MAP_FILE)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return mapping, live


if __name__ == "__main__":
    identify_cameras()
