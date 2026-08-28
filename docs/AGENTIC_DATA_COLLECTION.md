# Agentic data collection for Pasteur

This pipeline keeps ACT in charge of continuous motion and adds a slow,
auditable supervisor at semantic checkpoints.  It is inspired by CaP-X's
multi-turn visual differencing and persistent skill library, but generated
Python is never executed directly on the real robot.

## Runtime contract

The supervised checkpoints are:

1. initial scene;
2. immediately before the policy closes the gripper;
3. after the object has been lifted;
4. immediately before release;
5. after a stable release.

The 30 Hz ACT/ConeE path does not wait on visual reasoning except at these
stops.  Head SAM runs only at a checkpoint.  Three-camera timestamps,
measured state, command provenance, masks, head depth, verifier decisions, and
operator interventions are saved under the run audit directory.

Closing the gripper and lifting are not sufficient evidence that the intended
object is attached. A fresh checkpoint SAM mask must move with the object by a
resolution-independent fraction of the image diagonal, or an injected
semantic provider must explicitly confirm attachment. Matching the successful
demo's finger opening is retained as a diagnostic but remains `uncertain` by
itself.

If deterministic geometry and checkpoint SAM are still uncertain, a caller
may inject a `semantic_provider` into `AgenticPolicySupervisor`. It is invoked
only while held at that checkpoint, receives observations but no robot RPC,
and may return semantic facts such as `target_attached` or
`target_supported`. It is never called for, and cannot override, a
deterministic rejection.

The normal petri cycle is:

```
incubator -> microscope -> bench -> incubator
```

`ring` and `no_ring` are session conditions, not separate hard-coded task
logic.  Reference pose tolerances are measured in petri-dish diameters rather
than pixels. At startup, every demonstrated release pose is checked against
the following task's demonstrated close pose. A discontinuity rejects the
profile before ACT can run, preventing a mislabeled or invented return leg
from entering the autonomous cycle.

## Modes

- `shadow`: receives policy actions and records the first proposed action, but
  never sends a robot command. Startup machine-zero/home motion is also
  disabled in this mode.
- `supervised`: one `--armed` authorization enables audited execution; any
  rejected or unresolved checkpoint holds the current pose.
- `auto`: the same gates as supervised mode, with automatic task cycling after
  a verified clean success.

Start the existing SAM service at the endpoint configured in
`src/configs/pasteur_agentic_petri_collection.json`, then run a shadow test:

```bash
python src/run_agentic_collection.py \
  --mode shadow --condition ring \
  --host PEACOCK_POLICY_SERVER
```

After reviewing the audit, explicitly authorize one live session:

```bash
python src/run_agentic_collection.py \
  --mode supervised --armed --condition ring \
  --host PEACOCK_POLICY_SERVER
```

The phone UI defaults to `http://PASTEUR_IP:8098/`.  It can hold the current
pose, terminate for teleop takeover, or override only an **uncertain** gate.
A rejected gate cannot be overridden.  An overridden episode is classified as
`recovery`, never `clean_success`.

## Dataset policy

Every attempt is retained with one of these classes:

- `clean_success`: all automatic predicates accepted; eligible for ACT BC;
- `recovery`: operator intervention or a promoted recovery skill was used;
- `failure`: a physical/task predicate was rejected;
- `uncertain`: evidence was insufficient;
- `invalid`: timing, camera, action-stream, or state-machine integrity failed.

Convert only clean agentic successes with:

```bash
python src/convert_to_lerobot.py \
  --data_dirs DIRS... --task_names TASKS... \
  --output_dir OUTPUT --repo_id REPO \
  --agentic_classes clean_success --require_agentic_sidecar
```

Legacy demonstrations remain convertible when the two agentic filter flags
are omitted.

## Skill promotion

Recovery traces are parameterized in an object-relative frame and stored in
the agentic skill registry.  A candidate cannot be promoted merely because it
appears repeatedly.  Promotion requires:

- at least five verified examples;
- at least two start-position bins;
- passing simulation and shadow validation;
- stable preconditions and automatically checkable postconditions.

Until a recovery skill is promoted, an abnormal checkpoint holds and requests
teleop rather than improvising an unverified real-robot motion.
