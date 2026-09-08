"""Calibrate the camera from an ArUco marker of known size in the bench photo.

One printed marker (or board) placed at a known spot fixes what a single uncalibrated
photo cannot: the camera pose, the metric scale, AND the world/robot-base frame (we take
the marker frame as the origin, so recovered object poses come out directly in a frame
the arm can use). Detection is real (OpenCV); intrinsics default to a FoV guess until a
proper checkerboard calibration is supplied.
"""
from __future__ import annotations

import numpy as np
import cv2

from wetrobo.perception.camera import Camera, K_from_fovy

# Compatibility API for old single-marker callers. New deployments use the multi-tag
# estimator in fiducials.py; keep the family consistent so old code cannot silently
# detect a different printed dictionary.
_DEF_DICT = cv2.aruco.DICT_APRILTAG_36H11


def generate_charuco_assets(out_dir, *, squares=(5, 7), square_length_m=0.03,
                            marker_length_m=0.015, dpi=300):
    """Generate the exact board consumed by :func:`calibrate_intrinsics_charuco`."""
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard(squares, square_length_m, marker_length_m, dictionary)
    width_m, height_m = squares[0] * square_length_m, squares[1] * square_length_m
    px = (round(width_m / 0.0254 * dpi), round(height_m / 0.0254 * dpi))
    bitmap = board.generateImage(px, marginSize=0, borderBits=1)
    png = out / "charuco_5x7_30mm.png"; cv2.imwrite(str(png), bitmap)
    pdf = out / "charuco_5x7_30mm_actual_size.pdf"
    with PdfPages(pdf) as pages:
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0.08, 0.20, width_m / 0.210, height_m / 0.297])
        ax.imshow(bitmap, cmap="gray", vmin=0, vmax=255); ax.axis("off")
        fig.text(0.08, 0.15, "ChArUco 5x7; square=30 mm; marker=15 mm; DICT_5X5_100")
        ruler = fig.add_axes([0.08, 0.08, 0.1 / 0.210, 0.025])
        ruler.plot([0, 100], [0, 0], "k", lw=2); ruler.set_xlim(0, 100)
        ruler.set_xticks([0, 50, 100]); ruler.set_yticks([])
        ruler.set_xlabel("100 mm verification ruler — print at 100%, no fit-to-page")
        pages.savefig(fig); plt.close(fig)
    return [png, pdf]


def calibrate_intrinsics_charuco(images: list[np.ndarray], camera_id: str, *,
                                 squares=(5, 7), square_length_m=0.03,
                                 marker_length_m=0.015):
    """Calibrate one camera from 20+ ChArUco views and return a CalibrationProfile.

    All images must have the same resolution. Views with fewer than six interpolated
    ChArUco corners are ignored; fewer than 15 usable views is rejected.
    """
    from wetrobo.perception.fiducials import CalibrationProfile
    if len(images) < 20:
        raise ValueError("capture at least 20 ChArUco views")
    h, w = images[0].shape[:2]
    if any(im.shape[:2] != (h, w) for im in images):
        raise ValueError("all calibration images must have identical resolution")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard(squares, square_length_m, marker_length_m, dictionary)
    detector = cv2.aruco.CharucoDetector(board)
    all_corners, all_ids = [], []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        cc, ci, _, _ = detector.detectBoard(gray)
        if cc is not None and len(cc) >= 6:
            all_corners.append(cc); all_ids.append(ci)
    if len(all_corners) < 15:
        raise ValueError(f"need at least 15 usable ChArUco views, got {len(all_corners)}")
    rms, K, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, (w, h), None, None)
    return CalibrationProfile.create(camera_id, w, h, K, dist, rms)


def calibrate_hand_eye(T_base_gripper: np.ndarray, T_camera_target: np.ndarray,
                       profile, *, max_validation_m=0.005, max_validation_deg=1.0):
    """Solve eye-in-hand calibration from paired robot FK and target observations.

    Inputs are N×4×4 ``T_base_gripper`` and ``T_camera_target``. The returned profile
    stores ``T_gripper_camera``. A deterministic 20% held-out consistency score gates
    the result; this is a geometric repeatability check, not a substitute for a real
    independent metrology target.
    """
    bg = np.asarray(T_base_gripper, float); ct = np.asarray(T_camera_target, float)
    if bg.shape != ct.shape or bg.ndim != 3 or bg.shape[1:] != (4, 4) or len(bg) < 20:
        raise ValueError("need at least 20 paired N×4×4 transforms")
    train = np.arange(len(bg)) % 5 != 0
    R, t = cv2.calibrateHandEye(list(bg[train, :3, :3]), list(bg[train, :3, 3]),
                                list(ct[train, :3, :3]), list(ct[train, :3, 3]),
                                method=cv2.CALIB_HAND_EYE_DANIILIDIS)
    T_gc = np.eye(4); T_gc[:3, :3] = R; T_gc[:3, 3] = np.asarray(t).reshape(3)
    # For a static target, T_base_target = T_base_gripper T_gripper_camera T_camera_target.
    bt = np.array([bg[i] @ T_gc @ ct[i] for i in range(len(bg))])
    centre = np.median(bt[train, :3, 3], axis=0)
    held = ~train
    err = float(np.max(np.linalg.norm(bt[held, :3, 3] - centre, axis=1)))
    from scipy.spatial.transform import Rotation
    mean_rotation = Rotation.from_matrix(bt[train, :3, :3]).mean()
    rot_err = float(np.max((mean_rotation.inv() *
                            Rotation.from_matrix(bt[held, :3, :3])).magnitude()) * 180 / np.pi)
    if err > max_validation_m:
        raise ValueError(f"hand-eye held-out error {err*1000:.1f}mm exceeds "
                         f"{max_validation_m*1000:.1f}mm")
    if rot_err > max_validation_deg:
        raise ValueError(f"hand-eye held-out rotation error {rot_err:.2f}deg exceeds "
                         f"{max_validation_deg:.2f}deg")
    from dataclasses import replace
    return replace(profile, T_mount_camera=T_gc.tolist(), mount_frame="gripper",
                   hand_eye_validation_m=err, hand_eye_validation_deg=rot_err)


def marker_object_points(marker_length_m: float) -> np.ndarray:
    """The 4 marker corners in the marker frame (z=0 plane), OpenCV corner order
    (TL, TR, BR, BL), centered at the marker origin."""
    s = marker_length_m / 2
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)


def detect_markers(image: np.ndarray, dict_id: int = _DEF_DICT):
    """Return {marker_id: (4,2) corner pixels}. Real OpenCV detection."""
    aru = cv2.aruco.getPredefinedDictionary(dict_id)
    detector = cv2.aruco.ArucoDetector(aru, cv2.aruco.DetectorParameters())
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    corners, ids, _ = detector.detectMarkers(gray)
    out = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            out[int(i)] = c.reshape(4, 2).astype(np.float64)
    return out


def calibrate_from_aruco(image: np.ndarray, marker_length_m: float,
                         origin_marker_id: int, dict_id: int = _DEF_DICT,
                         K: np.ndarray | None = None,
                         dist=np.zeros(5)) -> tuple[Camera, dict]:
    """Recover a metric `Camera` (world frame = the origin marker's frame) from the photo.

    Returns (camera, detections). Raises if the origin marker is not found."""
    H, W = image.shape[:2]
    if K is None:
        K = K_from_fovy(50.0, W, H)          # provisional intrinsics (see module doc)
    dets = detect_markers(image, dict_id)
    if origin_marker_id not in dets:
        raise ValueError(f"origin marker {origin_marker_id} not detected "
                         f"(found {sorted(dets)})")
    objp = marker_object_points(marker_length_m)
    ok, rvec, tvec = cv2.solvePnP(objp, dets[origin_marker_id].astype(np.float32),
                                  K.astype(np.float32), dist,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise RuntimeError("solvePnP failed on origin marker")
    R_wc, _ = cv2.Rodrigues(rvec)
    return Camera(K=K, R_wc=R_wc, t_wc=tvec.reshape(3), width=W, height=H), dets
