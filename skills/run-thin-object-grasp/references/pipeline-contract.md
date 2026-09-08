# Thin-object primitive contract

## Ownership

Codex or a human chooses the next semantic primitive. Deterministic code owns
all timing-sensitive work within that primitive. One exclusive lease protects
the physical right-arm controller. The left arm receives zero commands.

Each primitive writes an immutable `result.json` containing the run ID,
sequence, primitive, profile hash, predecessor hash, measurements, evidence,
whether commands were sent, and `allowed_next_actions`. A decision contains
the exact result hash. Reject stale, edited, skipped, or cross-run decisions.

## Primitive gates

| Primitive | Required evidence | Nominal successor |
| --- | --- | --- |
| `observe` | Fresh head/right image; target identity and scene freshness | `plan-hover` |
| `plan-hover` | Current target pose; homing start; MuJoCo collision audit | `move-hover` |
| `move-hover` | Measured endpoint convergence; no unexpected contact | `align-hover` |
| `align-hover` | Fresh right view; shape/rim in grasp window; level approach | `descend-bottom` |
| `descend-bottom` | Normal vertical descent completed; XY/orientation fixed | `seat-2mm` |
| `seat-2mm` | Exactly one extra 2 mm request; fresh image; measured level | `close` |
| `close` | One continuous close; calibrated non-empty obstruction | `verify-lift` |
| `verify-lift` | Straight lift; persistent obstruction; same target follows | `home` |
| `recover` | Vertical clearance before open/retreat | `observe` or `home` |
| `home` | Measured home convergence and open gripper | `observe` or `stop` |

Rejected observation or planning sends no arm command. Rejected contact-height
evidence permits only recovery/stop. `seat-2mm` is single-shot and may never be
entered from itself or retried without returning through recovery and a new
normal descent.

## Target identity

Use semantic shape/rim consistency as primary identity. For a Petri lid this
means a SAM mask whose geometry is compatible with a thin circular object,
with temporal continuity and reviewed ROI context. Use the printed blue cross
for local continuity when visible. Use fixed AprilTags for camera/metric scene
registration when useful. Neither optional cue may authorize closure if the
shape target, tool frame, or grasp window is absent.

Avoid global largest-area or fixed-pixel decisions. Normalize areas and errors
by image/tool scale and physical calibration. Preserve target-like distractor,
occlusion, border, lighting, microscope/arm confusion, and transparent-rim
examples as replay tests.

## Height and closure

Normal descent reaches the audited support/contact endpoint first. The final
seating primitive then requests `0.002 m` down once while holding XY,
orientation, and open aperture. It is not a correction loop and is not folded
into normal descent. Recapture after the command and require measured jaw level
and target-in-window evidence before closure.

Close once continuously. Classify aperture against calibrated empty and
non-empty populations. Lift 2 mm first, verify persistent obstruction, then
complete the vertical lift and require the same target to follow.

## Failure routing

| Failure | Required route |
| --- | --- |
| Target/registration stale | `observe` again or `stop`; no motion |
| MuJoCo route rejected | repair scene/calibration; no motion |
| Hover target visible but offset | bounded free-space correction |
| Low target/tool absent or misaligned | vertical `recover`, then retry |
| Jaw level rejected | vertical `recover`; never correct XY at contact |
| Final seating execution or recapture rejected | vertical `recover`; do not repeat 2 mm |
| Empty closure | vertical `recover`; record failure |
| Target fails to follow lift | place/recover; record failure |

## Promotion

Treat runtime and pending alignments as acceleration only. They never authorize
closure. Promote a stable calibration only after complete grasp, initial lift,
full lift, and target-follow verification. Require consecutive end-to-end
successes before handing the same transition policy to a Codexless sequencer.
