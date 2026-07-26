# CLAUDE.md — WetRobo

Guidance for Claude Code (and any agent) working in this repository. This file
documents **WetRobo**, the code-based autonomous wet-lab robot built under `wetrobo/`.
For the legacy Pi0.5 VLA teleop/training/inference pipeline, see `README.md`.

## What WetRobo is

WetRobo replaces the learned VLA policy with a **code-based** agent that learns wet-lab
tasks by *performing* them in a MuJoCo simulation: perceive → act → verify the real
end-state → reflect on the failure → refine → retry. "Learning" is the accumulation and
refinement of **skill parameters** (and, optionally, skill code), not network weights.

This machine does not drive the real robot, so everything here is **sim-first**. The
first task is **flask → incubator** with the right 5-DOF Piper arm.

### Operational thesis: SAM-calibrated digital twin

The checked-in MJCF is a **nominal geometry prior**, not a measurement of today's lab.
Runtime object and fixture poses come from synchronized RGB-D plus instance-preserving
SAM masks, transformed through an explicitly registered camera-to-robot calibration.
If that transform, capture synchronization, or perception quality gate is missing, the
observation remains camera-local and must not update MuJoCo poses or collision state.

`daily_cad_ablation` is a legacy simulation experiment and may still be useful for
controlled research. Its oracle-CAD and synthetic MuJoCo segmentation conditions are
not evidence for the real SAM calibration path and must be reported separately.

## Architecture (`wetrobo/`)

| Module | Role |
|---|---|
| `sim/ik.py` | `ArmIK` — mink differential IK (quadprog) on the shared model for one arm. |
| `sim/lab_env.py` | `LabEnv` — headless MuJoCo world; grasp/release/move_ee/render/step. |
| `perception/cad.py` | `CADObserver` — the day's CAD: exact poses (ground truth) as a `BenchState`. |
| `perception/vision.py` | `VisionObserver` — renders depth+segmentation and back-projects; models transparent-glass depth failure (dropout + correlated bias). |
| `perception/sam.py` | Quality-gated saved SAM+RGB-D calibration; only registered observations may become robot-base poses. |
| `perception/fiducials.py` | Versioned AprilTag manifests, exact-size assets, multi-anchor camera pose, and hard quality gates. |
| `skills/library.py` | `SkillLibrary` — parametric `pick`/`place` skills + JSON param store. |
| `tasks/flask_to_incubator.py` | Goal region derived from the incubator geoms; success = real flask pose inside it. |
| `agent/planner.py` | `DeterministicPlanner` — the reproducible attempt/verify/reflect/refine loop (runs the ablation). |
| `agent/llm_agent.py` | `LLMSkillAuthor` — **opt-in** LLM proposer of parameter patches; never required. |
| `experiment/` | `layouts.py` (daily flask positions) + `daily_cad_ablation.py` (the A/B). |
| `episode_log.py` | `EpisodeLog` — numpy-safe JSONL writer/reader for every attempt + episode. |
| `report.py` | Aggregates a JSONL log into the table + figure. |
| `run_wetrobo.py` | Run a single episode (CAD or vision). |
| `fiducial_cli.py` | Intrinsic/hand-eye calibration, mount registration, validation, and measured daily-CAD authoring. |

Reused from `robot/piper-mujoco/bench_verify/`: `scene_graph` (Item/BenchState, Kabsch),
`verify` (SceneVerifier), `mujoco_oracle` (ground truth, render, perturb). The world is
`robot/piper-mujoco/xml/lab-scene.xml`.

### Hybrid controller

- **DeterministicPlanner** runs the reproducible ablation. Its "learning" is fixed rules
  that map a failure outcome to a parameter nudge, persisted in the `SkillLibrary`.
- **LLMSkillAuthor** is the escalation path: given the *real* logged failures, Claude
  proposes a parameter patch which is **clamped to safe physical ranges** before it
  touches the sim. It is gated by `available()` (needs the `anthropic` SDK + a key), so
  the ablation is fully reproducible without it.

## Running it

```bash
# one episode (independent variable = --observer)
python -m wetrobo.run_wetrobo --observer cad
python -m wetrobo.run_wetrobo --observer vision --seed 3 --max-attempts 6

# the daily-CAD A/B (real rollouts -> JSONL)
python -m wetrobo.experiment.daily_cad_ablation --days 6 --seeds 4 --out runs/ablation.jsonl

# figure + table from the log
python -m wetrobo.report --log runs/ablation.jsonl --out runs/figure.png

# opt-in LLM skill-author worked example (summarises real logged failures)
python -m wetrobo.agent.llm_agent --log runs/ablation.jsonl --condition vision
```

## Data-integrity rules (highest priority)

- **Never fabricate data.** Every number in the paper comes from a real MuJoCo rollout
  logged to JSONL. The no-CAD baseline runs **real** perception on rendered RGB-D — it is
  not a hobbled constant. Geometry and goal regions are derived from the MJCF, never
  hardcoded. `bench_verify/validate.py` is a labeled pre-registration harness, not evidence.
- **If an LLM proposes anything, the sim still decides.** `LLMSkillAuthor` only proposes
  numbers for existing knobs; success is always the real `SceneVerifier` end-state.
- **SAM is the runtime pose authority; CAD supplies shape priors.** A nominal MJCF pose
  never overrides an accepted SAM+RGB-D observation. AprilTags are optional providers
  of the camera-to-robot transform, not object-localization requirements. Object-tag
  localization remains a separate `tag_assisted_deployment` condition.
- **Fail closed across coordinate frames.** Camera-local or support-plane-levelled
  observations may be displayed beside the nominal model, but they cannot update a
  robot-base `BenchState`, MuJoCo body pose, or collision map until an accepted explicit
  transform is present.
- **No implicit tag transforms.** An unregistered `T_parent_tag` is unusable. Never use
  identity as a fallback or infer metric size from an image filename.

## Documented sim abstractions (kept honest, not hidden)

These are deliberate modelling choices for a weak 5-DOF arm; each is commented at its
definition so results are not over-claimed:

- **Sponge-pad kinematic grasp** (`LabEnv.grasp`): the object attaches only if its true
  grasp point actually sits **between the two inner jaw pads** — projecting onto the
  segment between the `pad_upper`/`pad_lower` sites (`|along|≤tol`, centred in the gap)
  and within `pad_len` of that axis (the pad, not the housing, contacts it). Measured in
  the gripper frame, so an off-axis or badly-oriented approach misses. This is what
  preserves the CAD-vs-vision signal under real physics — vision's transparent-depth
  error now misses the pads instead of sliding through a loose proximity check.
- **5-DOF side-approach orientation** (`APPROACH_QUAT`): the arm's 6th wrist joint is
  welded, so a vertical top-down grasp is **kinematically unreachable** at the bench's
  reach distances (measured 18–60 cm short). Every reach instead commands one achievable
  shallow side-approach (top-down pitched −50°) whose horizontal jaws straddle the
  vertical flask neck; the arm holds it consistently at pick and place (Δtilt ~1°).
- **Freeze-during-reach**: a bench object is frozen while the gripper descends, so a
  single IK step can't sweep it sideways; it unfreezes at the grasp gate.
- **Held-object collisions disabled during carry**, restored on release.
- **Runtime arm gain stiffening** (`_stiffen_arms`): the position actuators are weak;
  gains are raised at reset so the arm reaches its target instead of falling ~6 cm short.
- **Incubator: solid walls, open top**: the chamber walls collide (the gripper cannot
  pass through the body); only the top is left open so the arm can load the shelf from
  above through the open door.
- **Transparent-glass depth model** (`VisionObserver`): dropout + correlated per-object
  bias, reproducing why depth sensing fails on the transparent flask.

## Conventions

- Keep new numbers derived from the MJCF or measured from rollouts — do not hardcode.
- Treat MJCF fixture and object poses as nominal unless their provenance says they were
  produced by an accepted calibration artifact.
- Preserve SAM instance IDs and synchronized RGB/depth/qpos provenance. Legacy blended
  overlays are acceptable for offline recovery, not the preferred real-time format.
- Log every attempt and episode through `EpisodeLog`; `report.py` reads only logs.
- Durable tests live under `test/`; delete scratch scripts when done.
