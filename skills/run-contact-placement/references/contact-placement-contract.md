# Contact placement contract

## Responsibility boundary

Deterministic code owns frame freshness, normalized RGB-D deprojection,
physical/model arm mapping, level-route construction, exact-CAD IK, MuJoCo
collision audit, 30 Hz streaming, pressure hold, measured progress, state
persistence, and the phone evidence page. Codex or an operator may choose a
semantic target, confirm an ambiguous target once, or select among named
recovery actions. Neither belongs in the servo loop.

## Authority order

1. Measured joint state plus exact robot/tool CAD controls robot geometry.
2. Fresh depth plus SAM or a confirmed normalized target controls moveable
   target and support geometry.
3. Multi-view RGB-D polygons or reviewed catalog geometry controls static
   collision objects.
4. Pressure residual or support-plane agreement controls contact claims.
5. Missing or contradictory evidence remains unknown and blocks motion.

Image position never determines the physical arm. The production model's
historically crossed branch names are an explicit adapter detail.

## Portable inputs

### Start JSON

```json
{
  "measured_q_physical_rad": [0, 0, 0, 0, 0, 0],
  "measured_pose_wxyz_xyz": [1, 0, 0, 0, 0.3, 0.0, 0.4]
}
```

Always acquire this from the current hardware session. Do not substitute home
unless the measured state independently confirms home.

### Goal JSON

A semantic-scene goal contains `semantic_name`, `position_robot_m`,
`support_normal_robot`, `characteristic_scale_m`, `source`, and
`scene_revision`.

An operator-confirmed RGB-D goal replaces `position_robot_m` with
`normalized_uv`, `depth_npy`, `intrinsics_fx_fy_cx_cy`, and
`camera_to_robot_4x4`. The local depth neighborhood is a fraction of image
shape, not a fixed pixel radius.

### Observation JSON

Every transition identifies `physical_arm`, `scene_revision`, and a frame list.
Each frame includes `camera`, unique `frame_id`, and `captured_at_s`. Contact
observations additionally report requested/measured descent, torque change,
support clearance, and whether the pressure guard latched. Release observations
report support overlap and object-on-support evidence.

## Transition gates

| Transition | Required evidence |
| --- | --- |
| Observe to approach | Fresh required cameras, unchanged scene revision, explicit physical arm |
| Approach to descend | Scale-normalized goal alignment and a motion-ready audited route |
| Descend to contact | Pressure latch, or repeated low progress plus torque/support agreement |
| Descend to rebranch | Low progress without torque/support agreement |
| Contact to release | Object remains over the intended support |
| Release to retract | Object remains on support after opening |
| Retract to complete | Fresh vertical-retract evidence |

Low progress by itself never means the lowest reachable physical point. It can
mean joint-limit saturation, an IK branch mismatch, dropped commands, or a
controller mode problem.

## Motion adapter invariants

- Begin from the measured q used by the plan; reject a changed start.
- Execute only `motion_ready=true` plans and returned named actions.
- Use larger scale-derived chunks in audited free space and smaller
  scale-derived probes near contact.
- Keep the jaw plane parallel to the support at the planned hover and check it
  again before the first descent probe.
- Keep the observer arm fixed unless a separately audited stopped-observer
  transition is selected.
- Enforce pressure safety during all hardware motion. Workspace absence from a
  demonstration is not a collision model; use current semantic geometry.
- Hold measured q on pressure trip, tracking error, stale camera, changed scene,
  or deadline failure.

## Other-lab onboarding

1. Reconstruct and review a semantic 3D scene.
2. Establish physical robot bases, exact CAD, tool frames, and camera-to-robot
   transforms without image-left/right assumptions.
3. Label supports and obstacles with SAM plus depth; replace incomplete observed
   surfaces with conservative completed collision geometry.
4. Calibrate support normals and tool span/axes.
5. Create a site profile and validate at least two starts, two goals, and two
   image resolutions in preview.
6. Run pressure-guarded empty-tool rehearsals before carrying an object.
7. Promote only immutable runs containing plans, observations, transitions,
   torque audit, and phone-published image hashes.

AprilTags may simplify metric registration but are not mandatory. If used,
their IDs and placement are site calibration, not task policy.
