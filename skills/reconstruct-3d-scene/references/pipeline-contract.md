# SAM-first scene pipeline contract

## Sources of authority

Use this order:

1. Synchronized joint state plus repository MJCF/CAD for articulated bodies.
2. Quality-gated SAM masks plus synchronized depth for movable objects.
3. Measured RGB-D polygons for visible static surfaces.
4. Catalog templates or conservative primitives for hidden object volume.
5. Unknown space remains unknown.

Never use unconstrained global ICP to reshape or independently rotate robot
links. Never apply MuJoCo `mesh_pos` or `mesh_quat` a second time to runtime
mesh vertices.

## Main implementation

- `src/build_semantic_scene.py`: orchestration CLI and artifact generation.
- `rollout/semantic_scene_pipeline.py`: catalog validation, mask ownership,
  support discovery, completion, confidence, unknown detection, and collision
  gates.
- `src/configs/scene_object_catalog.json`: semantic prompts and geometry
  policies.
- `src/daily_scene_ui.py`: low-confidence phone confirmation.
- `docs/AUTOMATIC_3D_SCENE_WORKFLOW.md`: complete operator and extension guide.

## Input

The organized mesh NPZ contains:

- `vertices_xyz_m`: `[height*width, 3]`, expressed in a levelled metric frame.
- `faces`: triangle indices over the organized grid.
- `valid_vertex_mask`: optional `[height*width]` depth-validity mask.

The profile declares `organized_shape_hw`, catalog location, thresholds,
camera/calibration identity, and whether camera-to-robot extrinsics have been
accepted. Catalog entries declare prompts, completion policy, primitive,
dimension prior/range, transparency, support policy, confidence threshold,
color, and optional model.

For articulated models, use `robot_placement.instances` to map model root
bodies to named calibration anchors. Recover base height from the associated
SAM component, apply one shared upright yaw, and save a derived positioned
MJCF. Missing base placement keeps collision and motion readiness false.

## Output and gates

`scene.json` is the canonical result. Each object records its SAM provenance,
measured mesh, inferred geometry, support, completion method, quality terms,
confidence, and status.

- `display_ready`: enough information exists for visual inspection.
- `collision_ready`: every object is accepted, MuJoCo compiles, and completed
  geometry passes intersection gates.
- `motion_ready`: collision-ready plus an explicitly accepted camera-to-robot
  transform.

Observed and inferred layers must remain separately toggleable. Mobile mesh
traces must stay below practical WebGL index limits.

## Extension rule

Add a new object to the JSON catalog before adding Python. Add Python only
when a genuinely new completion family is needed. Test it with synthetic
RGB-D, one low-confidence case, one support/occlusion case, and a saved real
capture.
