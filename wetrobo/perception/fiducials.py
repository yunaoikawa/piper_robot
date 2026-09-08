"""Versioned fiducial assets, calibration profiles, and quality-gated pose estimation.

Operational tags use OpenCV's AprilTag 36h11 dictionary.  A manifest is the source of
truth for printed size and mounting transforms; image filenames are never interpreted
as calibration data.  Poses follow ``T_parent_child`` notation and homogeneous 4x4
matrices, avoiding the camera/world direction ambiguity common in ad-hoc calibration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


FAMILY = "DICT_APRILTAG_36H11"
FAMILY_ID = cv2.aruco.DICT_APRILTAG_36H11


def _matrix(value, *, optional=False):
    if value is None and optional:
        return None
    a = np.asarray(value, float)
    if a.shape != (4, 4) or not np.allclose(a[3], [0, 0, 0, 1]):
        raise ValueError("transform must be a 4x4 homogeneous matrix")
    return a


@dataclass(frozen=True)
class MarkerSpec:
    marker_id: int
    role: str                         # "anchor" | "object"
    size_m: float                     # measured outer black-square edge
    parent: str                       # "robot_base" or catalog container id
    T_parent_tag: list[list[float]] | None = None
    printed_size_m: float | None = None
    note: str = ""

    def validate(self) -> None:
        if self.role not in {"anchor", "object"}:
            raise ValueError(f"marker {self.marker_id}: invalid role {self.role!r}")
        if self.size_m <= 0:
            raise ValueError(f"marker {self.marker_id}: size_m must be positive")
        if self.printed_size_m is not None and self.printed_size_m <= 0:
            raise ValueError(f"marker {self.marker_id}: printed_size_m must be positive")
        _matrix(self.T_parent_tag, optional=True)

    @property
    def calibrated_size_m(self) -> float:
        return float(self.printed_size_m or self.size_m)


@dataclass
class MarkerManifest:
    family: str = FAMILY
    version: int = 1
    markers: list[MarkerSpec] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "MarkerManifest":
        raw = json.loads(Path(path).read_text())
        obj = cls(raw.get("family", FAMILY), int(raw.get("version", 1)),
                  [MarkerSpec(**m) for m in raw["markers"]])
        obj.validate()
        return obj

    def save(self, path: str | Path) -> Path:
        self.validate()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return p

    def validate(self) -> None:
        if self.family != FAMILY:
            raise ValueError(f"unsupported family {self.family!r}; expected {FAMILY}")
        ids = [m.marker_id for m in self.markers]
        if len(ids) != len(set(ids)):
            raise ValueError("marker ids must be unique")
        for m in self.markers:
            m.validate()

    def by_id(self) -> dict[int, MarkerSpec]:
        return {m.marker_id: m for m in self.markers}

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class CalibrationProfile:
    camera_id: str
    width: int
    height: int
    K: list[list[float]]
    dist: list[float]
    calibration_id: str
    created_at: str
    intrinsic_rms_px: float
    T_mount_camera: list[list[float]] | None = None
    mount_frame: str | None = None
    hand_eye_validation_m: float | None = None
    hand_eye_validation_deg: float | None = None

    @classmethod
    def create(cls, camera_id, width, height, K, dist, rms, **kwargs):
        now = datetime.now(timezone.utc).isoformat()
        seed = f"{camera_id}:{now}:{np.asarray(K).tolist()}"
        return cls(camera_id, int(width), int(height), np.asarray(K).tolist(),
                   np.asarray(dist).reshape(-1).tolist(), hashlib.sha256(seed.encode()).hexdigest()[:16],
                   now, float(rms), **kwargs)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationProfile":
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> Path:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return p

    def validate_for(self, image: np.ndarray, max_rms_px=0.5, max_age_days=180) -> None:
        h, w = image.shape[:2]
        if (w, h) != (self.width, self.height):
            raise ValueError(f"image is {w}x{h}, profile is {self.width}x{self.height}")
        if self.intrinsic_rms_px > max_rms_px:
            raise ValueError(f"intrinsic RMS {self.intrinsic_rms_px:.3f}px exceeds {max_rms_px}px")
        created = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86400
        if age_days > max_age_days:
            raise ValueError(f"calibration is {age_days:.0f} days old; recalibrate")


@dataclass
class PoseQuality:
    accepted: bool
    reason: str
    visible_anchor_ids: list[int]
    reprojection_rms_px: float | None
    workplane_error_m: float | None
    T_camera_robot: list[list[float]] | None


def tag_corners(size_m: float) -> np.ndarray:
    s = size_m / 2
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)


def transform_points(T, points):
    p = np.asarray(points, float)
    return (np.asarray(T, float)[:3, :3] @ p.T).T + np.asarray(T, float)[:3, 3]


def detect_tags(image: np.ndarray) -> dict[int, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(FAMILY_ID), params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64) for c, i in zip(corners, ids.flat)}


def estimate_camera_pose(image, profile: CalibrationProfile, manifest: MarkerManifest,
                         *, min_anchors=3, max_reprojection_px=1.5,
                         max_workplane_error_m=0.005) -> PoseQuality:
    """Estimate ``T_camera_robot`` from all registered anchor corners and gate quality.

    The work-plane error is leave-one-anchor-out: solve on the other anchors, intersect
    the omitted marker centre ray with its known Z plane, and compare in robot XY.
    This exposes layouts that have a low pixel residual but poor metric conditioning.
    """
    try:
        profile.validate_for(image)
    except ValueError as exc:
        return PoseQuality(False, str(exc), [], None, None, None)
    detections = detect_tags(image)
    anchors = [m for m in manifest.markers if m.role == "anchor" and
               m.T_parent_tag is not None and m.marker_id in detections]
    ids = [m.marker_id for m in anchors]
    if len(anchors) < min_anchors:
        return PoseQuality(False, f"need {min_anchors} registered anchors, found {len(anchors)}",
                           ids, None, None, None)
    centres = np.array([_matrix(m.T_parent_tag)[:3, 3] for m in anchors])
    planar_singular = np.linalg.svd(centres[:, :2] - centres[:, :2].mean(0),
                                    compute_uv=False)
    if len(planar_singular) < 2 or planar_singular[1] < 0.05:
        return PoseQuality(False, "anchor layout is collinear or spans less than 50 mm",
                           ids, None, None, None)
    obj = np.concatenate([transform_points(_matrix(m.T_parent_tag), tag_corners(m.calibrated_size_m))
                          for m in anchors]).astype(np.float32)
    img = np.concatenate([detections[m.marker_id] for m in anchors]).astype(np.float32)
    K, dist = np.asarray(profile.K), np.asarray(profile.dist)
    ok, rvec, tvec, _ = cv2.solvePnPRansac(obj, img, K, dist,
                                           flags=cv2.SOLVEPNP_ITERATIVE,
                                           reprojectionError=max_reprojection_px * 2)
    if not ok:
        return PoseQuality(False, "solvePnPRansac failed", ids, None, None, None)
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - img) ** 2, axis=1))))
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tvec.reshape(3)
    metric_errors = []
    for omit in anchors:
        keep = [m for m in anchors if m.marker_id != omit.marker_id]
        if len(keep) < 3:
            continue
        oo = np.concatenate([transform_points(_matrix(m.T_parent_tag), tag_corners(m.calibrated_size_m))
                             for m in keep]).astype(np.float32)
        ii = np.concatenate([detections[m.marker_id] for m in keep]).astype(np.float32)
        ok2, rv2, tv2 = cv2.solvePnP(oo, ii, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok2:
            continue
        R2, _ = cv2.Rodrigues(rv2); C = -R2.T @ tv2.reshape(3)
        uv = detections[omit.marker_id].mean(0)
        ray_c = np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0])
        ray_w = R2.T @ ray_c
        true = _matrix(omit.T_parent_tag)[:3, 3]
        if abs(ray_w[2]) > 1e-9:
            hit = C + ((true[2] - C[2]) / ray_w[2]) * ray_w
            metric_errors.append(float(np.linalg.norm(hit[:2] - true[:2])))
    anchor_cam = (R @ centres.T + tvec.reshape(3, 1)).T
    metres_per_pixel_error = rms * float(np.mean(anchor_cam[:, 2])) / float(
        (K[0, 0] + K[1, 1]) / 2)
    # Four anchors give a stronger leave-one-out estimate. With exactly three, retain
    # the conservative residual-to-work-plane conversion after rejecting collinearity.
    world_err = max([metres_per_pixel_error, *metric_errors])
    accepted = rms <= max_reprojection_px and world_err <= max_workplane_error_m
    reason = "ok" if accepted else (
        f"reprojection {rms:.3f}px exceeds {max_reprojection_px}px" if rms > max_reprojection_px
        else f"work-plane error {world_err*1000:.1f}mm exceeds {max_workplane_error_m*1000:.1f}mm")
    return PoseQuality(accepted, reason, ids, rms, world_err, T.tolist())


def estimate_registered_objects(image, profile: CalibrationProfile, manifest: MarkerManifest,
                                quality: PoseQuality) -> list[dict]:
    """Recover registered object poses in robot base; never assumes an identity mount.

    Returns auditable pose records. Object markers with an unregistered
    ``T_object_tag`` are omitted instead of silently treating tag centre as object centre.
    """
    if not quality.accepted or quality.T_camera_robot is None:
        raise ValueError(f"camera pose rejected: {quality.reason}")
    detections = detect_tags(image)
    K, dist = np.asarray(profile.K), np.asarray(profile.dist)
    T_cr = np.asarray(quality.T_camera_robot)
    T_rc = np.linalg.inv(T_cr)
    out = []
    for marker in manifest.markers:
        if marker.role != "object" or marker.T_parent_tag is None:
            continue
        corners = detections.get(marker.marker_id)
        if corners is None:
            continue
        ok, rv, tv = cv2.solvePnP(tag_corners(marker.calibrated_size_m),
                                  corners.astype(np.float32), K, dist,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        R_ct, _ = cv2.Rodrigues(rv)
        T_ct = np.eye(4); T_ct[:3, :3] = R_ct; T_ct[:3, 3] = tv.reshape(3)
        T_rt = T_rc @ T_ct
        T_ot = _matrix(marker.T_parent_tag)
        T_ro = T_rt @ np.linalg.inv(T_ot)
        projected, _ = cv2.projectPoints(tag_corners(marker.calibrated_size_m), rv, tv, K, dist)
        rms = float(np.sqrt(np.mean(np.sum(
            (projected.reshape(-1, 2) - corners) ** 2, axis=1))))
        out.append({"container": marker.parent, "marker_id": marker.marker_id,
                    "T_robot_object": T_ro.tolist(), "tag_reprojection_rms_px": rms})
    return out


def bounded_local_correction(T_robot_object_cad, T_robot_object_observed, *,
                             max_translation_m=0.015, max_rotation_deg=5.0) -> np.ndarray:
    """Accept a wrist-camera correction only when it is plausibly local.

    Larger discrepancies mean the scene or calibration changed and must trigger daily
    CAD regeneration rather than silently dragging a grasp target across the bench.
    """
    cad = _matrix(T_robot_object_cad); observed = _matrix(T_robot_object_observed)
    delta = np.linalg.inv(cad) @ observed
    translation = float(np.linalg.norm(delta[:3, 3]))
    angle = float(np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))))
    if translation > max_translation_m or angle > max_rotation_deg:
        raise ValueError(f"scene changed: correction={translation*1000:.1f}mm/{angle:.1f}deg; "
                         "regenerate daily CAD")
    return observed.copy()


def generate_marker_assets(manifest: MarkerManifest, out_dir: str | Path,
                           dpi: int = 300) -> list[Path]:
    """Generate exact-size PNG and SVG files plus a contact-sheet PDF with 100 mm ruler."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(FAMILY_ID)
    made = []
    for m in manifest.markers:
        px = max(256, int(round(m.size_m / 0.0254 * dpi)))
        marker_bitmap = cv2.aruco.generateImageMarker(dictionary, m.marker_id, px)
        # One module of white quiet-zone around the 6x6 code + black border. The
        # declared size remains the outer BLACK edge, which solvePnP expects.
        margin = max(1, round(px / 8))
        bitmap = cv2.copyMakeBorder(marker_bitmap, margin, margin, margin, margin,
                                    cv2.BORDER_CONSTANT, value=255)
        png = out / f"tag_{m.marker_id:03d}_{round(m.size_m*1000)}mm.png"
        cv2.imwrite(str(png), bitmap); made.append(png)
        # SVG embeds the canonical bitmap and explicit physical dimensions.
        ok, encoded = cv2.imencode(".png", bitmap)
        import base64
        href = base64.b64encode(encoded).decode()
        mm = m.size_m * 1000
        total_mm = mm * bitmap.shape[0] / px
        svg = out / f"tag_{m.marker_id:03d}_{round(mm)}mm.svg"
        svg.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_mm}mm" height="{total_mm}mm" '
                       f'viewBox="0 0 {bitmap.shape[0]} {bitmap.shape[0]}"><image width="{bitmap.shape[0]}" height="{bitmap.shape[0]}" '
                       f'href="data:image/png;base64,{href}"/></svg>\n')
        made.append(svg)
    pdf = out / "print_sheet_actual_size.pdf"
    with PdfPages(pdf) as pages:
        for m in manifest.markers:
            bitmap = cv2.imread(str(out / f"tag_{m.marker_id:03d}_{round(m.size_m*1000)}mm.png"), 0)
            fig = plt.figure(figsize=(8.27, 11.69))
            total_size = m.size_m * 1.25  # 6x6 payload + 1 black + 1 white module/side
            ax = fig.add_axes([0.12, 0.35, total_size / 0.210, total_size / 0.297])
            ax.imshow(bitmap, cmap="gray", vmin=0, vmax=255); ax.axis("off")
            fig.text(0.12, 0.30, f"{FAMILY} id={m.marker_id} black edge={m.size_m*1000:.1f} mm")
            ruler = fig.add_axes([0.12, 0.18, 0.1 / 0.210, 0.03]); ruler.plot([0, 100], [0, 0], "k", lw=2)
            ruler.set_xlim(0, 100); ruler.set_xticks([0, 50, 100]); ruler.set_yticks([])
            ruler.set_xlabel("100 mm verification ruler — print at 100%, no fit-to-page")
            pages.savefig(fig); plt.close(fig)
    made.append(pdf)
    manifest.save(out / "manifest.json"); made.append(out / "manifest.json")
    return made
