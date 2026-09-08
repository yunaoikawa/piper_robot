"""Quality-gated SAM + RGB-D calibration for the nominal MuJoCo lab.

The MJCF contributes geometry and articulation only.  Object poses in this
module are derived exclusively from a synchronized semantic RGB-D artifact and
an explicit camera-to-robot transform.  Camera-local observations may be
inspected, but cannot become a robot-base ``BenchState``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

import wetrobo._paths  # noqa: F401 - exposes bench_verify on sys.path
from bench_verify.scene_graph import BenchState
from rollout.scene_semantics import (
    LABEL_LID,
    LABEL_NAMES,
    LABEL_ROBOT,
)
from wetrobo.perception.catalog import LabwareCatalog


class CalibrationRejected(RuntimeError):
    """Raised when an observation is not safe to publish in robot coordinates."""


@dataclass(frozen=True)
class SamQualityGates:
    max_rgb_depth_delta_s: float = 0.050
    min_registration_inliers: int = 30
    max_registration_residual_px: float = 1.5
    max_support_rms_m: float = 0.010
    min_support_inlier_fraction: float = 0.25
    min_label_vertices: int = 20
    max_mask_overlap_fraction: float = 0.20
    max_shape_relative_error: float = 0.35
    min_sam_score: float = 0.75
    required_labels: tuple[int, ...] = (LABEL_ROBOT, LABEL_LID)


@dataclass(frozen=True)
class SamQuality:
    accepted: bool
    issues: tuple[str, ...]
    metrics: dict


@dataclass(frozen=True)
class SamFootprint:
    semantic_label: str
    semantic_label_id: int
    frame: str
    center_m: np.ndarray
    rotation: np.ndarray
    extents_m: np.ndarray
    point_count: int
    yaw_observable: bool
    estimation_method: str

    def matrix(self) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = self.rotation
        transform[:3, 3] = self.center_m
        return transform

    def to_dict(self) -> dict:
        return {
            "semantic_label": self.semantic_label,
            "semantic_label_id": self.semantic_label_id,
            "frame": self.frame,
            "center_m": self.center_m.tolist(),
            "rotation": self.rotation.tolist(),
            "extents_m": self.extents_m.tolist(),
            "point_count": self.point_count,
            "yaw_observable": self.yaw_observable,
            "estimation_method": self.estimation_method,
        }


@dataclass(frozen=True)
class SamLabelBinding:
    """Bind one SAM instance/label to a closed-set catalog entry."""

    semantic_label: str | int
    instance_id: str
    container: str
    sam_score: float | None = None


@dataclass(frozen=True)
class SamBenchCalibration:
    bench_state: BenchState
    footprints: dict[str, SamFootprint]
    T_robot_level: np.ndarray
    provenance: dict


def _rigid_transform(value, *, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4):
        raise CalibrationRejected(f"{name} must be a 4x4 transform")
    if not np.all(np.isfinite(transform)):
        raise CalibrationRejected(f"{name} contains non-finite values")
    if not np.allclose(transform[3], (0, 0, 0, 1), atol=1e-8):
        raise CalibrationRejected(f"{name} has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5):
        raise CalibrationRejected(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise CalibrationRejected(f"{name} rotation is not proper")
    return transform


def _transform_points(transform, points) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return points @ transform[:3, :3].T + transform[:3, 3]


def _semantic_id(label: str | int) -> int:
    if isinstance(label, (int, np.integer)):
        label_id = int(label)
        if label_id not in LABEL_NAMES:
            raise KeyError(f"unknown semantic label id {label_id}")
        return label_id
    inverse = {name: label_id for label_id, name in LABEL_NAMES.items()}
    if label not in inverse:
        raise KeyError(f"unknown semantic label name {label!r}")
    return inverse[label]


@dataclass
class SamCalibrationArtifact:
    directory: Path
    report: dict
    vertices_level_m: np.ndarray
    faces: np.ndarray
    semantic_labels: np.ndarray
    semantic_labels_rgb: np.ndarray | None
    rgb_camera_matrix: np.ndarray | None
    T_level_camera: np.ndarray
    quality: SamQuality
    gates: SamQualityGates
    provenance: dict

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        gates: SamQualityGates | None = None,
    ) -> "SamCalibrationArtifact":
        directory = Path(directory)
        gates = gates or SamQualityGates()
        report_path = directory / "esdf_report.json"
        mesh_path = directory / "scene_mesh_levelled.npz"
        volume_path = directory / "scene_esdf.npz"
        label_image_path = directory / "semantic_labels_rgb.npy"
        missing = [
            str(path)
            for path in (report_path, mesh_path, volume_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "SAM calibration artifact is incomplete: " + ", ".join(missing)
            )

        report = json.loads(report_path.read_text())
        with np.load(mesh_path) as mesh:
            vertices = np.asarray(mesh["vertices_xyz_m"], dtype=float)
            faces = np.asarray(mesh["faces"], dtype=np.int32)
            labels = np.asarray(mesh["semantic_labels"], dtype=np.uint8)
        with np.load(volume_path) as volume:
            rotation = np.asarray(
                volume["camera_to_level_rotation"], dtype=float
            )
            translation = np.asarray(
                volume["camera_to_level_translation"], dtype=float
            )
        label_image = (
            np.asarray(np.load(label_image_path), dtype=np.uint8)
            if label_image_path.is_file()
            else None
        )
        camera_matrix_value = report.get("rgb_camera_matrix")
        rgb_camera_matrix = (
            np.asarray(camera_matrix_value, dtype=float)
            if camera_matrix_value is not None
            else None
        )

        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("scene vertices must have shape (N, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("scene faces must have shape (M, 3)")
        if labels.shape != (len(vertices),):
            raise ValueError("semantic labels must match scene vertices")
        if len(faces) and (np.min(faces) < 0 or np.max(faces) >= len(vertices)):
            raise ValueError("scene faces reference invalid vertices")
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("scene_esdf has an invalid camera-to-level transform")
        if label_image is not None and label_image.ndim != 2:
            raise ValueError("semantic_labels_rgb must be a 2D label image")
        if (
            rgb_camera_matrix is not None
            and rgb_camera_matrix.shape != (3, 3)
        ):
            raise ValueError("rgb_camera_matrix must have shape (3, 3)")
        T_level_camera = np.eye(4)
        T_level_camera[:3, :3] = rotation
        T_level_camera[:3, 3] = translation
        _rigid_transform(T_level_camera, name="T_level_camera")

        issues: list[str] = []
        metrics: dict = {}
        if not report.get("sam_semantics", False):
            issues.append("artifact has no SAM semantics")

        sync_delta = float(report.get("rgb_depth_file_mtime_delta_s", np.inf))
        metrics["rgb_depth_delta_s"] = sync_delta
        if not np.isfinite(sync_delta) or sync_delta > gates.max_rgb_depth_delta_s:
            issues.append(
                f"RGB-depth delta {sync_delta:.6f}s exceeds "
                f"{gates.max_rgb_depth_delta_s:.6f}s"
            )

        registration = report.get("sam_registration")
        metrics["sam_registration_present"] = registration is not None
        if registration is not None:
            inliers = int(registration.get("inliers", 0))
            residual = float(
                registration.get("median_inlier_residual_px", np.inf)
            )
            metrics["registration_inliers"] = inliers
            metrics["registration_residual_px"] = residual
            if inliers < gates.min_registration_inliers:
                issues.append(
                    f"SAM registration has {inliers} inliers; "
                    f"need {gates.min_registration_inliers}"
                )
            if (
                not np.isfinite(residual)
                or residual > gates.max_registration_residual_px
            ):
                issues.append(
                    f"SAM registration residual {residual:.3f}px exceeds "
                    f"{gates.max_registration_residual_px:.3f}px"
                )

        coordinate_frame = report.get("coordinate_frame")
        metrics["coordinate_frame"] = coordinate_frame
        if coordinate_frame != "support-plane-levelled":
            issues.append(
                "horizontal SAM calibration requires support-plane-levelled geometry"
            )
        if label_image is None:
            issues.append(
                "semantic_labels_rgb.npy is required for transparent target pose"
            )
        if rgb_camera_matrix is None:
            issues.append(
                "rgb_camera_matrix provenance is required for mask ray projection"
            )
        if (
            label_image is not None
            and report.get("rgb_shape_hw") is not None
            and tuple(report["rgb_shape_hw"]) != label_image.shape
        ):
            issues.append(
                "semantic label image shape does not match RGB provenance"
            )

        support = report.get("support_plane_fit")
        if support is not None:
            support_rms = float(support.get("rms_m", np.inf))
            support_inliers = float(support.get("inlier_fraction", 0.0))
            metrics["support_rms_m"] = support_rms
            metrics["support_inlier_fraction"] = support_inliers
            if (
                not np.isfinite(support_rms)
                or support_rms > gates.max_support_rms_m
            ):
                issues.append(
                    f"support RMS {support_rms:.4f}m exceeds "
                    f"{gates.max_support_rms_m:.4f}m"
                )
            if support_inliers < gates.min_support_inlier_fraction:
                issues.append(
                    f"support inlier fraction {support_inliers:.3f} is below "
                    f"{gates.min_support_inlier_fraction:.3f}"
                )
        else:
            metrics["support_fit_source"] = "external_or_unrecorded"
            issues.append("support-plane fit quality is unavailable")

        label_vertex_counts = {
            LABEL_NAMES[label_id]: int(np.count_nonzero(labels == label_id))
            for label_id in gates.required_labels
        }
        metrics["label_vertex_counts"] = label_vertex_counts
        for label_id in gates.required_labels:
            count = label_vertex_counts[LABEL_NAMES[label_id]]
            if count < gates.min_label_vertices:
                issues.append(
                    f"{LABEL_NAMES[label_id]} has {count} vertices; "
                    f"need {gates.min_label_vertices}"
                )

        overlap = int(report.get("sam_mask_overlap_pixels", 0))
        surface_pixels = report.get("semantic_surface_pixels", {})
        dynamic_counts = [
            int(surface_pixels.get(LABEL_NAMES[label_id], 0))
            for label_id in gates.required_labels
        ]
        overlap_denominator = max(1, min(dynamic_counts, default=1))
        overlap_fraction = overlap / overlap_denominator
        metrics["sam_mask_overlap_pixels"] = overlap
        metrics["sam_mask_overlap_fraction_of_smaller_mask"] = overlap_fraction
        if overlap_fraction > gates.max_mask_overlap_fraction:
            issues.append(
                f"SAM mask overlap fraction {overlap_fraction:.3f} exceeds "
                f"{gates.max_mask_overlap_fraction:.3f}"
            )

        provenance = {
            "artifact_version": report.get("artifact_version", 1),
            "pose_authority": "quality_gated_sam_plus_synchronized_rgbd",
            "nominal_mjcf_role": "geometry_prior_only",
            "rgb_source": report.get("rgb_source"),
            "depth_source": report.get("depth_source"),
            "coordinate_frame": coordinate_frame,
            "sam_scores_available": bool(
                report.get("sam_scores_available", False)
            ),
            "sam_score_status": (
                "recorded"
                if report.get("sam_scores_available", False)
                else "unavailable_in_legacy_overlay"
            ),
            "capture_sync_verified": bool(
                report.get("capture_sync_verified", False)
            ),
            "capture_sync_evidence": report.get(
                "capture_sync_evidence", "unrecorded"
            ),
            "camera_to_robot_source": None,
        }
        quality = SamQuality(not issues, tuple(issues), metrics)
        return cls(
            directory=directory,
            report=report,
            vertices_level_m=vertices,
            faces=faces,
            semantic_labels=labels,
            semantic_labels_rgb=label_image,
            rgb_camera_matrix=rgb_camera_matrix,
            T_level_camera=T_level_camera,
            quality=quality,
            gates=gates,
            provenance=provenance,
        )

    def require_quality(self) -> None:
        if not self.quality.accepted:
            raise CalibrationRejected(
                "SAM artifact rejected: " + "; ".join(self.quality.issues)
            )

    def compute_T_robot_level(self, T_robot_camera) -> np.ndarray:
        """Compose robot<-level from an explicit robot<-camera transform."""

        self.require_quality()
        if T_robot_camera is None:
            raise CalibrationRejected(
                "T_robot_camera is required; camera-local SAM cannot update MuJoCo"
            )
        robot_camera = _rigid_transform(
            T_robot_camera, name="T_robot_camera"
        )
        return _rigid_transform(
            robot_camera @ np.linalg.inv(self.T_level_camera),
            name="T_robot_level",
        )

    def label_points(self, label: str | int) -> np.ndarray:
        label_id = _semantic_id(label)
        return self.vertices_level_m[self.semantic_labels == label_id].copy()

    def project_label_to_support_plane(
        self, label: str | int, *, maximum_points: int = 20000
    ) -> np.ndarray:
        """Intersect SAM mask rays with z=0 in the levelled support frame.

        This is the preferred localization for transparent planar targets. It
        uses the SAM silhouette and camera geometry, not unreliable depth
        returns from transparent plastic.
        """

        self.require_quality()
        label_id = _semantic_id(label)
        if self.semantic_labels_rgb is None or self.rgb_camera_matrix is None:
            raise CalibrationRejected(
                "support-plane projection needs semantic_labels_rgb.npy and "
                "rgb_camera_matrix provenance"
            )
        yy, xx = np.where(self.semantic_labels_rgb == label_id)
        if len(xx) < self.gates.min_label_vertices:
            raise CalibrationRejected(
                f"{LABEL_NAMES[label_id]} has too few mask pixels"
            )
        if len(xx) > maximum_points:
            indices = np.linspace(
                0, len(xx) - 1, maximum_points, dtype=np.int64
            )
            xx, yy = xx[indices], yy[indices]
        pixels = np.column_stack(
            (xx.astype(float) + 0.5, yy.astype(float) + 0.5, np.ones(len(xx)))
        )
        rays_camera = pixels @ np.linalg.inv(self.rgb_camera_matrix).T
        rotation = self.T_level_camera[:3, :3]
        origin = self.T_level_camera[:3, 3]
        rays_level = rays_camera @ rotation.T
        denominator = rays_level[:, 2]
        scale = np.full(len(rays_level), np.nan, dtype=float)
        usable = np.abs(denominator) > 1e-9
        scale[usable] = -origin[2] / denominator[usable]
        usable &= scale > 0
        points = origin + scale[usable, None] * rays_level[usable]
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) < self.gates.min_label_vertices:
            raise CalibrationRejected(
                f"{LABEL_NAMES[label_id]} rays do not intersect the support plane"
            )
        points[:, 2] = 0.0
        return points

    def estimate_horizontal_footprint(
        self,
        label: str | int,
        *,
        T_robot_level=None,
        trim_quantile: float = 0.02,
    ) -> SamFootprint:
        """Estimate a robust horizontal footprint without consulting CAD pose."""

        self.require_quality()
        label_id = _semantic_id(label)
        use_support_projection = (
            label_id == LABEL_LID
            and self.semantic_labels_rgb is not None
            and self.rgb_camera_matrix is not None
        )
        if use_support_projection:
            points = self.project_label_to_support_plane(label_id)
            estimation_method = "sam_mask_support_plane_intersection"
        else:
            points = self.label_points(label_id)
            estimation_method = "labelled_depth_surface"
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) < self.gates.min_label_vertices:
            raise CalibrationRejected(
                f"{LABEL_NAMES[label_id]} has too few finite points"
            )
        q = float(trim_quantile)
        if q < 0 or q >= 0.25:
            raise ValueError("trim_quantile must be in [0, 0.25)")
        lower = np.quantile(points, q, axis=0)
        upper = np.quantile(points, 1.0 - q, axis=0)
        inside = np.all((points >= lower) & (points <= upper), axis=1)
        trimmed = points[inside]
        if len(trimmed) < self.gates.min_label_vertices:
            trimmed = points

        center = np.median(trimmed, axis=0)
        xy = trimmed[:, :2] - center[:2]
        covariance = xy.T @ xy / max(1, len(xy) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        axis_x = eigenvectors[:, order[0]]
        if axis_x[0] < 0:
            axis_x = -axis_x
        axis_y = np.array([-axis_x[1], axis_x[0]])
        rotation_level = np.eye(3)
        rotation_level[:2, 0] = axis_x
        rotation_level[:2, 1] = axis_y
        local_xy = xy @ rotation_level[:2, :2]
        xy_extent = (
            np.quantile(local_xy, 1.0 - q, axis=0)
            - np.quantile(local_xy, q, axis=0)
        )
        z_extent = float(
            np.quantile(trimmed[:, 2], 1.0 - q)
            - np.quantile(trimmed[:, 2], q)
        )
        ratio = float(
            eigenvalues[1] / max(eigenvalues[0], np.finfo(float).eps)
        )
        yaw_observable = ratio < 0.80
        if label_id == LABEL_LID:
            # A circular lid has no physical yaw. Mask clipping/perspective can
            # make PCA appear directional, but that direction is not a pose.
            rotation_level = np.eye(3)
            yaw_observable = False

        frame = "support-plane-levelled"
        rotation = rotation_level
        if T_robot_level is not None:
            transform = _rigid_transform(
                T_robot_level, name="T_robot_level"
            )
            center = _transform_points(transform, center[None])[0]
            rotation = transform[:3, :3] @ rotation_level
            frame = "robot_base"
        return SamFootprint(
            semantic_label=LABEL_NAMES[label_id],
            semantic_label_id=label_id,
            frame=frame,
            center_m=center,
            rotation=rotation,
            extents_m=np.array([xy_extent[0], xy_extent[1], z_extent]),
            point_count=int(len(trimmed)),
            yaw_observable=yaw_observable,
            estimation_method=estimation_method,
        )

    def calibrate(
        self,
        bindings: Iterable[SamLabelBinding],
        catalog: LabwareCatalog,
        T_robot_camera,
    ) -> SamBenchCalibration:
        """Publish accepted SAM instances as a robot-base ``BenchState``."""

        T_robot_level = self.compute_T_robot_level(T_robot_camera)
        if not self.provenance["capture_sync_verified"]:
            raise CalibrationRejected(
                "capture synchronization is not timestamp-verified; "
                "file mtime is diagnostic evidence only"
            )
        items = []
        footprints: dict[str, SamFootprint] = {}
        binding_provenance = []
        seen_labels: set[int] = set()
        for binding in bindings:
            label_id = _semantic_id(binding.semantic_label)
            if label_id in seen_labels:
                raise CalibrationRejected(
                    f"{LABEL_NAMES[label_id]} is bound more than once; "
                    "this artifact stores class masks, not separate instances"
                )
            seen_labels.add(label_id)
            if binding.sam_score is None:
                raise CalibrationRejected(
                    f"{binding.instance_id}: SAM score is missing; "
                    "legacy overlays are diagnostic-only"
                )
            score = float(binding.sam_score)
            if score < self.gates.min_sam_score or score > 1.0:
                raise CalibrationRejected(
                    f"{binding.instance_id}: SAM score {score:.3f} is outside "
                    f"the accepted range [{self.gates.min_sam_score:.3f}, 1]"
                )
            entry = catalog.get(binding.container)
            level_footprint = self.estimate_horizontal_footprint(
                binding.semantic_label
            )
            expected_xy = None
            if entry.shape == "cyl":
                expected_xy = np.repeat(2.0 * float(entry.dims[0]), 2)
            elif entry.shape == "box":
                expected_xy = np.asarray(entry.dims[:2], dtype=float)
            shape_relative_error = None
            if expected_xy is not None:
                observed_xy = np.sort(level_footprint.extents_m[:2])
                expected_xy = np.sort(expected_xy)
                shape_relative_error = np.abs(
                    observed_xy - expected_xy
                ) / np.maximum(expected_xy, 1e-9)
                if np.max(shape_relative_error) > (
                    self.gates.max_shape_relative_error
                ):
                    raise CalibrationRejected(
                        f"{binding.instance_id}: SAM footprint "
                        f"{observed_xy.tolist()}m does not match catalog "
                        f"{expected_xy.tolist()}m"
                    )
            center_level = level_footprint.center_m.copy()
            rotation_level = level_footprint.rotation.copy()
            if entry.shape == "cyl":
                center_level[2] = float(entry.dims[1]) / 2.0
                # A circular footprint has no observable yaw, even if mask
                # clipping makes its PCA eigenvalues slightly anisotropic.
                rotation_level = np.eye(3)
            elif entry.shape == "box":
                center_level[2] = float(entry.dims[2]) / 2.0
            center_robot = _transform_points(
                T_robot_level, center_level[None]
            )[0]
            rotation_robot = (
                T_robot_level[:3, :3] @ rotation_level
            )
            footprint = SamFootprint(
                semantic_label=level_footprint.semantic_label,
                semantic_label_id=level_footprint.semantic_label_id,
                frame="robot_base",
                center_m=center_robot,
                rotation=rotation_robot,
                extents_m=level_footprint.extents_m,
                point_count=level_footprint.point_count,
                yaw_observable=(
                    level_footprint.yaw_observable
                    if entry.shape != "cyl"
                    else False
                ),
                estimation_method=level_footprint.estimation_method,
            )
            footprints[binding.instance_id] = footprint
            items.append(
                catalog.to_item(
                    binding.container,
                    binding.instance_id,
                    center_robot,
                    rotation_robot,
                    score,
                )
            )
            binding_provenance.append(
                {
                    **asdict(binding),
                    "semantic_label": level_footprint.semantic_label,
                    "geometry_source": "closed_set_catalog_prior",
                    "geometry_provisional": bool(entry.provisional_dims),
                    "shape_relative_error": (
                        None
                        if shape_relative_error is None
                        else shape_relative_error.tolist()
                    ),
                    "pose_source": "sam_rgbd",
                }
            )
        provenance = {
            **self.provenance,
            "camera_to_robot_source": "explicit_T_robot_camera",
            "T_robot_level": T_robot_level.tolist(),
            "bindings": binding_provenance,
        }
        state = BenchState(
            f"sam_calibrated_{self.directory.name}",
            items,
            frame="robot_base",
            captured_by="wetrobo.perception.sam",
        )
        return SamBenchCalibration(
            bench_state=state,
            footprints=footprints,
            T_robot_level=T_robot_level,
            provenance=provenance,
        )

    def to_bench_state(
        self,
        bindings: Iterable[SamLabelBinding],
        catalog: LabwareCatalog,
        T_robot_camera,
    ) -> BenchState:
        return self.calibrate(
            bindings, catalog, T_robot_camera
        ).bench_state

    def summary(self) -> dict:
        return {
            "directory": str(self.directory),
            "quality": {
                "accepted": self.quality.accepted,
                "issues": list(self.quality.issues),
                "metrics": self.quality.metrics,
            },
            "gates": asdict(self.gates),
            "provenance": self.provenance,
            "T_level_camera": self.T_level_camera.tolist(),
        }
