"""Deterministic SAM-first RGB-D scene reconstruction primitives.

The module intentionally contains no robot RPC and no language-model call.
It turns semantic masks plus an organized, levelled RGB-D mesh into measured
surfaces, conservative completed geometry, provenance, and readiness gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt


SCHEMA = "piper_robot.semantic_scene/v1"
CATALOG_SCHEMA = "piper_robot.scene_object_catalog/v1"
PROFILE_SCHEMA = "piper_robot.semantic_scene_profile/v1"


def _finite_number(value, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _vector(value, length: int, name: str, *, positive=False) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {length} finite numbers")
    if positive and np.any(result <= 0.0):
        raise ValueError(f"{name} must be positive")
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CatalogObject:
    name: str
    prompts: tuple[str, ...]
    completion: str
    primitive: str | None
    nominal_size_m: tuple[float, float, float] | None
    size_range_m: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ] | None
    transparent: bool
    minimum_confidence: float
    support: str
    color: str
    model: str | None

    @classmethod
    def from_dict(cls, value: dict) -> "CatalogObject":
        name = str(value["name"]).strip()
        if not name:
            raise ValueError("catalog object name cannot be empty")
        prompts = tuple(str(item).strip() for item in value.get("prompts", ()))
        if not prompts or any(not item for item in prompts):
            raise ValueError(f"{name}.prompts must not be empty")
        completion = str(value.get("completion", "primitive"))
        if completion not in {"primitive", "template", "exact_cad", "observed_mesh"}:
            raise ValueError(f"{name}.completion is unsupported")
        primitive = value.get("primitive")
        if primitive is not None and primitive not in {"box", "cylinder"}:
            raise ValueError(f"{name}.primitive is unsupported")
        nominal = value.get("nominal_size_m")
        nominal_size = (
            None
            if nominal is None
            else tuple(_vector(nominal, 3, f"{name}.nominal_size_m", positive=True))
        )
        raw_range = value.get("size_range_m")
        size_range = None
        if raw_range is not None:
            if not isinstance(raw_range, list) or len(raw_range) != 2:
                raise ValueError(f"{name}.size_range_m must contain [minimum, maximum]")
            low = _vector(raw_range[0], 3, f"{name}.size_range_m[0]", positive=True)
            high = _vector(raw_range[1], 3, f"{name}.size_range_m[1]", positive=True)
            if np.any(low > high):
                raise ValueError(f"{name}.size_range_m minimum exceeds maximum")
            size_range = (tuple(low), tuple(high))
        threshold = _finite_number(
            value.get("minimum_confidence", 0.72),
            f"{name}.minimum_confidence",
        )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"{name}.minimum_confidence must be in [0, 1]")
        return cls(
            name=name,
            prompts=prompts,
            completion=completion,
            primitive=primitive,
            nominal_size_m=nominal_size,
            size_range_m=size_range,
            transparent=bool(value.get("transparent", False)),
            minimum_confidence=threshold,
            support=str(value.get("support", "horizontal_surface")),
            color=str(value.get("color", "#8b9bb4")),
            model=None if value.get("model") is None else str(value["model"]),
        )


def load_profile(path: str | Path) -> tuple[dict, dict[str, CatalogObject]]:
    profile_path = Path(path)
    profile = json.loads(profile_path.read_text())
    if profile.get("schema") != PROFILE_SCHEMA:
        raise ValueError(f"{profile_path}: unsupported profile schema")
    catalog_path = Path(profile["catalog"])
    if not catalog_path.is_absolute():
        catalog_path = (profile_path.parent / catalog_path).resolve()
    catalog_payload = json.loads(catalog_path.read_text())
    if catalog_payload.get("schema") != CATALOG_SCHEMA:
        raise ValueError(f"{catalog_path}: unsupported catalog schema")
    def resolve_existing(reference: str, base: Path) -> str:
        candidate = Path(reference)
        if candidate.is_absolute():
            return str(candidate)
        for parent in (base, *base.parents):
            resolved = (parent / candidate).resolve()
            if resolved.exists():
                return str(resolved)
        return str((base / candidate).resolve())

    object_payloads = []
    for raw in catalog_payload["objects"]:
        item = dict(raw)
        if item.get("model"):
            item["model"] = resolve_existing(item["model"], catalog_path.parent)
        object_payloads.append(item)
    objects = [CatalogObject.from_dict(item) for item in object_payloads]
    catalog = {item.name: item for item in objects}
    if len(catalog) != len(objects):
        raise ValueError(f"{catalog_path}: duplicate object names")
    profile["catalog_path"] = str(catalog_path)
    if profile.get("robot_model"):
        profile["robot_model"] = resolve_existing(
            profile["robot_model"], profile_path.parent
        )
    return profile, catalog


@dataclass(frozen=True)
class MaskObservation:
    instance_id: str
    semantic_name: str
    prompt: str
    mask_path: str
    sam_score: float
    model: str
    inference_ms: float


@dataclass(frozen=True)
class ObjectGeometry:
    kind: str
    center_xyz_m: tuple[float, float, float]
    size_xyz_m: tuple[float, float, float]
    yaw_rad: float


def load_organized_mesh(path: str | Path) -> dict[str, np.ndarray]:
    archive = np.load(path)
    required = {"vertices_xyz_m", "faces"}
    missing = required - set(archive.files)
    if missing:
        raise ValueError(f"{path}: missing arrays {sorted(missing)}")
    vertices = np.asarray(archive["vertices_xyz_m"], dtype=float)
    faces = np.asarray(archive["faces"], dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices_xyz_m must have shape [N,3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [M,3]")
    valid = (
        np.asarray(archive["valid_vertex_mask"], dtype=bool)
        if "valid_vertex_mask" in archive.files
        else np.all(np.isfinite(vertices), axis=1)
    )
    if valid.shape != (len(vertices),):
        raise ValueError("valid_vertex_mask shape does not match vertices")
    return {"vertices": vertices, "faces": faces, "valid": valid}


def load_mask(path: str | Path, shape_hw: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    if image.shape != shape_hw:
        image = cv2.resize(
            image,
            (shape_hw[1], shape_hw[0]),
            interpolation=cv2.INTER_AREA,
        )
    return image > 100


def exclusive_masks(
    observations: Iterable[MaskObservation],
    shape_hw: tuple[int, int],
    *,
    transparent_semantics: Iterable[str] = (),
    nested_containment_threshold: float = 0.90,
    nested_score_margin: float = 0.05,
) -> list[tuple[MaskObservation, np.ndarray]]:
    """Resolve overlaps deterministically by confidence then semantic name.

    SAM can return the same opaque object under two prompts.  Merely assigning
    the overlap to the higher-confidence mask leaves a thin, misleading
    fragment of the lower-confidence class.  Suppress that whole candidate
    when it is almost contained by a stronger, differently-labelled opaque
    mask.  Transparent semantics are exempt because physically distinct
    objects such as a Petri dish and its lid legitimately overlap in image
    space.
    """

    loaded = [
        (item, load_mask(item.mask_path, shape_hw)) for item in observations
    ]
    transparent = set(transparent_semantics)
    suppressed: set[int] = set()
    for candidate_index, (candidate, candidate_mask) in enumerate(loaded):
        candidate_area = int(np.count_nonzero(candidate_mask))
        if candidate_area == 0:
            continue
        for owner_index, (owner, owner_mask) in enumerate(loaded):
            if (
                candidate_index == owner_index
                or candidate.semantic_name == owner.semantic_name
                or candidate.semantic_name in transparent
                or owner.semantic_name in transparent
                or owner.sam_score
                < candidate.sam_score + nested_score_margin
            ):
                continue
            containment = (
                np.count_nonzero(candidate_mask & owner_mask) / candidate_area
            )
            if containment >= nested_containment_threshold:
                suppressed.add(candidate_index)
                break
    loaded = [
        pair for index, pair in enumerate(loaded) if index not in suppressed
    ]
    # Confidence is authoritative. For equal-confidence accepted masks, a
    # smaller mask is the more-specific semantic region (for example a bottle
    # label inside a robot-shaped false positive) and owns the overlap.
    loaded.sort(
        key=lambda pair: (
            pair[0].sam_score,
            -int(np.count_nonzero(pair[1])),
            pair[0].semantic_name,
        )
    )
    owned: list[tuple[MaskObservation, np.ndarray]] = []
    claimed = np.zeros(shape_hw, dtype=bool)
    for item, mask in reversed(loaded):
        owned.append((item, mask & ~claimed))
        claimed |= mask
    owned.reverse()
    return owned


def robust_oriented_geometry(
    points: np.ndarray,
    *,
    catalog: CatalogObject | None,
    support_height_m: float | None,
) -> ObjectGeometry:
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        raise ValueError("at least three depth points are required")
    center_xy = np.median(points[:, :2], axis=0)
    xy = points[:, :2] - center_xy
    _, _, vh = np.linalg.svd(xy, full_matrices=False)
    yaw = float(np.arctan2(vh[0, 1], vh[0, 0]))
    rotation = np.array(
        [[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]
    )
    local_xy = (rotation @ xy.T).T
    low_xy, high_xy = np.quantile(local_xy, [0.02, 0.98], axis=0)
    low_z, high_z = np.quantile(points[:, 2], [0.02, 0.98])
    measured_size = np.array(
        [high_xy[0] - low_xy[0], high_xy[1] - low_xy[1], high_z - low_z]
    )
    measured_size = np.maximum(measured_size, 0.002)
    size = measured_size.copy()
    if catalog is not None and catalog.nominal_size_m is not None:
        nominal = np.asarray(catalog.nominal_size_m)
        # Transparent and strongly occluded objects use their catalog shape.
        if catalog.transparent or measured_size[2] < 0.15 * nominal[2]:
            size = nominal.copy()
        else:
            size = 0.65 * measured_size + 0.35 * nominal
    if catalog is not None and catalog.size_range_m is not None:
        minimum, maximum = (np.asarray(item) for item in catalog.size_range_m)
        size = np.clip(size, minimum, maximum)
    center_z = float((low_z + high_z) / 2)
    if support_height_m is not None:
        center_z = float(support_height_m + size[2] / 2)
    return ObjectGeometry(
        kind=(catalog.primitive if catalog and catalog.primitive else "box"),
        center_xyz_m=(float(center_xy[0]), float(center_xy[1]), center_z),
        size_xyz_m=tuple(float(item) for item in size),
        yaw_rad=yaw,
    )


def _organized_shape(vertex_count: int, image_shape: tuple[int, int]) -> None:
    if int(np.prod(image_shape)) != vertex_count:
        raise ValueError(
            f"mesh vertex count {vertex_count} does not match image {image_shape}"
        )


def discover_supports(
    vertices: np.ndarray,
    valid: np.ndarray,
    shape_hw: tuple[int, int],
    *,
    height_tolerance_m: float,
    minimum_area_fraction: float,
) -> list[dict]:
    """Find broad level surfaces without semantic or pixel-position priors."""

    _organized_shape(len(vertices), shape_hw)
    points = vertices.reshape(*shape_hw, 3)
    valid_image = valid.reshape(shape_hw)
    normal_ok = np.zeros(shape_hw, dtype=bool)
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    normal = np.cross(dx, dy)
    length = np.linalg.norm(normal, axis=-1)
    usable = length > 1e-8
    normal[usable] /= length[usable, None]
    normal_ok[1:-1, 1:-1] = usable & (
        np.abs(normal[..., 2]) >= np.cos(np.deg2rad(15.0))
    )
    horizontal = valid_image & normal_ok
    heights = points[..., 2][horizontal]
    if len(heights) < 3:
        return []
    bin_m = max(height_tolerance_m / 2.0, 0.001)
    low, high = np.quantile(heights, [0.01, 0.99])
    edges = np.arange(np.floor(low / bin_m) * bin_m, high + 2 * bin_m, bin_m)
    counts, _ = np.histogram(heights, bins=edges)
    candidates = np.argsort(counts)[::-1]
    supports = []
    claimed = np.zeros(shape_hw, dtype=bool)
    minimum_pixels = max(9, int(np.prod(shape_hw) * minimum_area_fraction))
    kernel = np.ones((3, 3), np.uint8)
    for index in candidates:
        if counts[index] < minimum_pixels:
            break
        height = float((edges[index] + edges[index + 1]) / 2)
        band = (
            horizontal
            & ~claimed
            & (np.abs(points[..., 2] - height) <= height_tolerance_m)
        )
        band = cv2.morphologyEx(
            band.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(band, 8)
        for component in range(1, count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area < minimum_pixels:
                continue
            mask = labels == component
            selected = points[mask]
            lower, upper = np.quantile(selected, [0.02, 0.98], axis=0)
            supports.append(
                {
                    "support_id": f"support-{len(supports) + 1}",
                    "height_m": float(np.median(selected[:, 2])),
                    "bounds_xy_m": [lower[:2].tolist(), upper[:2].tolist()],
                    "area_pixels": area,
                    "mask": mask,
                }
            )
            claimed |= mask
        if len(supports) >= 12:
            break
    supports.sort(key=lambda item: item["area_pixels"], reverse=True)
    return supports


def support_collision_boxes(
    support: dict,
    *,
    vertices: np.ndarray,
    valid: np.ndarray,
    shape_hw: tuple[int, int],
    tile_size_px: int = 8,
    minimum_points: int = 4,
) -> list[dict]:
    """Approximate an observed support mask without filling its occluded holes.

    A single world-space AABB is unsafe here: an arm standing through a cut-out
    or an occluded gap disappears when the non-rectangular RGB-D support mask
    is replaced by its bounding rectangle.  Small image-space tiles retain
    those holes while remaining cheap box collision geometry for MuJoCo/ESDF.
    """

    _organized_shape(len(vertices), shape_hw)
    tile = int(tile_size_px)
    if tile < 2:
        raise ValueError("tile_size_px must be at least 2")
    minimum = int(minimum_points)
    if minimum < 1:
        raise ValueError("minimum_points must be positive")
    mask = np.asarray(support["mask"], dtype=bool)
    if mask.shape != shape_hw:
        raise ValueError("support mask shape does not match organized RGB-D")
    points = np.asarray(vertices, dtype=float).reshape(*shape_hw, 3)
    valid_image = np.asarray(valid, dtype=bool).reshape(shape_hw)
    usable = mask & valid_image
    rows, columns = np.nonzero(usable)
    if not len(rows):
        return []
    result = []
    row_start = int(rows.min() // tile * tile)
    column_start = int(columns.min() // tile * tile)
    for top in range(row_start, int(rows.max()) + 1, tile):
        for left in range(column_start, int(columns.max()) + 1, tile):
            selected_mask = usable[top : top + tile, left : left + tile]
            if int(np.count_nonzero(selected_mask)) < minimum:
                continue
            selected = points[top : top + tile, left : left + tile][
                selected_mask
            ]
            lower, upper = np.quantile(selected[:, :2], [0.02, 0.98], axis=0)
            size = upper - lower
            if np.any(~np.isfinite(size)) or np.any(size <= 1e-5):
                continue
            result.append(
                {
                    "center_xy_m": ((lower + upper) / 2).tolist(),
                    "size_xy_m": np.maximum(size, 0.002).tolist(),
                    "source_pixel_bounds_xyxy": [
                        left,
                        top,
                        min(left + tile, shape_hw[1]),
                        min(top + tile, shape_hw[0]),
                    ],
                }
            )
    return result


def choose_support(points: np.ndarray, supports: list[dict]) -> dict | None:
    center = np.median(points, axis=0)
    lower_object, upper_object = np.quantile(points[:, :2], [0.05, 0.95], axis=0)
    object_span = upper_object - lower_object
    maximum_extrapolation = max(0.02, 0.5 * float(np.max(object_span)))
    candidates = []
    for support in supports:
        lower, upper = np.asarray(support["bounds_xy_m"], dtype=float)
        delta = np.maximum(np.maximum(lower - center[:2], center[:2] - upper), 0)
        xy_distance = float(np.linalg.norm(delta))
        if xy_distance <= maximum_extrapolation:
            candidates.append((xy_distance, support))
    if not candidates:
        return None
    # Occlusion, reflection, and transparent surfaces can put observed depth
    # on the far side of the support plane. XY containment is more reliable
    # than requiring every observed point to lie above it.
    median_z = float(np.median(points[:, 2]))
    return min(
        candidates,
        key=lambda item: (
            item[0],
            abs(float(item[1]["height_m"]) - median_z),
        ),
    )[1]


def quality_score(
    *,
    sam_score: float,
    mask: np.ndarray,
    valid_depth: np.ndarray,
    geometry: ObjectGeometry,
    catalog: CatalogObject | None,
    support_found: bool,
) -> tuple[float, dict]:
    mask_pixels = int(np.count_nonzero(mask))
    valid_pixels = int(np.count_nonzero(mask & valid_depth))
    depth_fraction = valid_pixels / max(1, mask_pixels)
    components, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    largest = (
        int(np.max(stats[1:, cv2.CC_STAT_AREA])) if components > 1 else 0
    )
    connected_fraction = largest / max(1, mask_pixels)
    size_score = 1.0
    if catalog is not None and catalog.size_range_m is not None:
        size = np.asarray(geometry.size_xyz_m)
        minimum, maximum = (np.asarray(item) for item in catalog.size_range_m)
        scale = np.maximum(maximum - minimum, 1e-6)
        violation = np.maximum(minimum - size, 0) + np.maximum(size - maximum, 0)
        size_score = float(np.exp(-4.0 * np.max(violation / scale)))
    support_score = (
        1.0
        if catalog is None or catalog.support == "none" or support_found
        else 0.25
    )
    weights = np.asarray([0.34, 0.28, 0.16, 0.12, 0.10])
    terms = np.asarray(
        [
            np.clip(sam_score, 0.0, 1.0),
            np.clip(depth_fraction, 0.0, 1.0),
            np.clip(connected_fraction, 0.0, 1.0),
            size_score,
            support_score,
        ]
    )
    score = float(weights @ terms)
    return score, {
        "sam_score": float(sam_score),
        "valid_depth_fraction": float(depth_fraction),
        "largest_component_fraction": float(connected_fraction),
        "size_prior_score": float(size_score),
        "support_score": float(support_score),
        "weights": weights.tolist(),
    }


def detect_unknown_objects(
    *,
    vertices: np.ndarray,
    valid: np.ndarray,
    shape_hw: tuple[int, int],
    claimed: np.ndarray,
    supports: list[dict],
    minimum_area_fraction: float,
) -> list[dict]:
    """Return interior residual RGB-D components above a discovered support."""

    points = vertices.reshape(*shape_hw, 3)
    candidate = valid.reshape(shape_hw) & ~claimed
    for support in supports:
        candidate &= ~cv2.dilate(
            support["mask"].astype(np.uint8), np.ones((5, 5), np.uint8)
        ).astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), 8
    )
    minimum_pixels = max(12, int(np.prod(shape_hw) * minimum_area_fraction))
    unknown = []
    height, width = shape_hw
    for component in range(1, count):
        x = int(stats[component, cv2.CC_STAT_LEFT])
        y = int(stats[component, cv2.CC_STAT_TOP])
        w = int(stats[component, cv2.CC_STAT_WIDTH])
        h = int(stats[component, cv2.CC_STAT_HEIGHT])
        area = int(stats[component, cv2.CC_STAT_AREA])
        touches_border = x == 0 or y == 0 or x + w == width or y + h == height
        if area < minimum_pixels or touches_border:
            continue
        mask = labels == component
        selected = points[mask]
        support = choose_support(selected, supports)
        if support is None:
            continue
        height_above = float(
            np.quantile(selected[:, 2], 0.90) - support["height_m"]
        )
        if height_above <= 0.01:
            continue
        unknown.append(
            {
                "instance_id": f"unknown-{len(unknown) + 1}",
                "mask": mask,
                "points": selected,
                "support": support,
                "area_pixels": area,
            }
        )
    return unknown


def aabb_intersections(objects: list[dict], tolerance_m: float = 0.002) -> list[dict]:
    intersections = []
    for index, first in enumerate(objects):
        if first.get("completion") == "exact_cad":
            continue
        first_geometry = first["geometry"]
        first_center = np.asarray(first_geometry["center_xyz_m"])
        first_half = np.asarray(first_geometry["size_xyz_m"]) / 2
        for second in objects[index + 1 :]:
            if second.get("completion") == "exact_cad":
                continue
            if (
                first.get("support_id") == second["instance_id"]
                or second.get("support_id") == first["instance_id"]
            ):
                continue
            second_geometry = second["geometry"]
            second_center = np.asarray(second_geometry["center_xyz_m"])
            second_half = np.asarray(second_geometry["size_xyz_m"]) / 2
            overlap = np.minimum(
                first_center + first_half, second_center + second_half
            ) - np.maximum(
                first_center - first_half, second_center - second_half
            )
            if np.all(overlap > tolerance_m):
                intersections.append(
                    {
                        "a": first["instance_id"],
                        "b": second["instance_id"],
                        "overlap_xyz_m": overlap.tolist(),
                    }
                )
    return intersections


def conservative_scene_esdf(
    *,
    vertices: np.ndarray,
    valid: np.ndarray,
    objects: list[dict],
    supports: list[dict],
    camera_origin_m: np.ndarray | None,
    voxel_size_m: float,
    maximum_voxels: int = 2_000_000,
) -> dict[str, np.ndarray | float | str]:
    """Voxelize observed rays and completed bodies without declaring unknown free."""

    points = np.asarray(vertices, dtype=float)[np.asarray(valid, dtype=bool)]
    if len(points) < 3:
        raise ValueError("ESDF requires at least three valid RGB-D points")
    lower, upper = np.quantile(points, [0.01, 0.99], axis=0)
    lower -= 0.08
    upper += 0.08
    voxel = float(voxel_size_m)
    if not np.isfinite(voxel) or voxel <= 0:
        raise ValueError("voxel_size_m must be finite and positive")
    shape = np.maximum(1, np.ceil((upper - lower) / voxel).astype(int) + 1)
    while int(np.prod(shape)) > int(maximum_voxels):
        voxel *= 1.2
        shape = np.maximum(1, np.ceil((upper - lower) / voxel).astype(int) + 1)
    observed = np.zeros(tuple(shape), dtype=bool)
    occupied = np.zeros(tuple(shape), dtype=bool)

    def indices(world):
        index = np.floor((np.asarray(world) - lower) / voxel).astype(int)
        return np.clip(index, 0, shape - 1)

    endpoint_indices = indices(points)
    occupied[tuple(endpoint_indices.T)] = True
    observed[tuple(endpoint_indices.T)] = True
    if camera_origin_m is not None:
        origin = _vector(camera_origin_m, 3, "camera_origin_m")
        stride = max(1, len(points) // 12000)
        for point in points[::stride]:
            length = float(np.linalg.norm(point - origin))
            count = max(2, int(np.ceil(length / voxel)))
            samples = origin + np.linspace(0.0, 0.98, count)[:, None] * (
                point - origin
            )
            ray_indices = indices(samples)
            observed[tuple(ray_indices.T)] = True

    completed = list(objects)
    for support in supports:
        support_thickness = 0.025
        boxes = support.get("collision_boxes")
        if not boxes:
            lower_xy, upper_xy = np.asarray(
                support["bounds_xy_m"], dtype=float
            )
            boxes = [
                {
                    "center_xy_m": ((lower_xy + upper_xy) / 2).tolist(),
                    "size_xy_m": (upper_xy - lower_xy).tolist(),
                }
            ]
        for box in boxes:
            completed.append(
                {
                    "geometry": {
                        "center_xyz_m": [
                            *box["center_xy_m"],
                            float(support["height_m"])
                            - support_thickness / 2,
                        ],
                        "size_xyz_m": [
                            *box["size_xy_m"],
                            support_thickness,
                        ],
                    }
                }
            )
    for record in completed:
        collision_boxes = record.get("collision_boxes")
        geometries = (
            [
                {
                    "center_xyz_m": item["center_xyz_m"],
                    "size_xyz_m": item["size_xyz_m"],
                }
                for item in collision_boxes
            ]
            if collision_boxes
            else [record["geometry"]]
        )
        for geometry in geometries:
            center = np.asarray(geometry["center_xyz_m"], dtype=float)
            half = np.asarray(geometry["size_xyz_m"], dtype=float) / 2
            low_index = indices(center - half)
            high_index = indices(center + half)
            slices = tuple(
                slice(int(low_index[axis]), int(high_index[axis]) + 1)
                for axis in range(3)
            )
            occupied[slices] = True
            observed[slices] = True
    distance = distance_transform_edt(~occupied) * voxel
    distance[occupied] = -voxel
    unknown = ~observed
    frontier = unknown & binary_dilation(observed) & ~occupied
    return {
        "schema": "piper_robot.conservative_scene_esdf/v1",
        "esdf_m": distance.astype(np.float32),
        "observed": observed,
        "occupied": occupied,
        "unknown_collision": unknown,
        "unknown_frontier": frontier,
        "origin_xyz_m": lower.astype(np.float64),
        "voxel_size_m": voxel,
    }


def scene_json_ready(value: object) -> object:
    """Remove runtime masks and normalize NumPy values for JSON output."""

    if isinstance(value, dict):
        return {
            key: scene_json_ready(item)
            for key, item in value.items()
            if key not in {"mask", "points"}
        }
    if isinstance(value, (list, tuple)):
        return [scene_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return scene_json_ready(asdict(value))
    return value
