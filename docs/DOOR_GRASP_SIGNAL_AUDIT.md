# Historical Door grasp signals

These figures show **measured gripper opening, not pressure**. The normalized
opening is dimensionless: 0 is fully closed and 1 fully open. It cannot be
converted into contact pressure or gripping force without additional sensing
and calibration. A blocked closing gripper does not prove correct handle contact.

![Per-trial phase comparison](assets/code_as_learning_machine/door_grasp_aperture_trials.png)

![Recorded observation blocks](assets/code_as_learning_machine/door_grasp_aperture_samples.png)

## Scope and provenance

The seven rows are physical pull attempts on August 8, 2026. T1--T7 are not
the paper's D1--D4 code configurations and are not the human teleoperation demos.
T1 is the first long demo-relative pull, T2--T5 retries 2--5, T6 the yaw-aligned
retry, and T7 the endpoint-verified autonomous opening. Three earlier saved
contact-only attempts are listed separately in the audit report. This is not
a claim of exhaustive coverage of every historical physical interaction.

The report records exact source paths and SHA-256 hashes. The extraction pairs
contact and proof records and checks that their contact poses agree. It does
not extract numerical data from prose or manufacture missing samples.

An audit of 466 JSON files in `data/runs/pasteur/incubator*20260808*/` found arm
joint-torque snapshots and torque-warning summaries, but no contact-pressure
or gripper-current time series. These torque data are not interchangeable with
gripper pressure. The representative teleop HDF5 also contains poses, gripper
opening, timestamps and depth, but no force/current channel.

## Reading the figures

- **Closed:** the controller's saved settled aperture, the median of the last
  10 of 70 monitoring samples after the close command.
- **After 5 mm pull:** median of 12 samples collected **after** the proof motion.
  These are not samples during the motion.
- **Post-pull:** one saved observation after the long pull or its interruption.
  For T7 this is the observation before recovery. T3 and T5 are missing from
  the selected numerical sources and remain missing, not zero-filled.
- Each raw-sample curve belongs to a separate observation block. Individual
  sample timestamps were not saved; the x-axis is sample index, not seconds.
  Delays between blocks are not compressed into a purported continuous trace.
- No continuous full-pull trace or complete checkpoint history is reconstructed.
  These figures cannot identify the exact time of slip or peak contact load.
- The dotted 0.02 line is the controller's empty-aperture reference, not a
  physical force limit or a sufficient test of grasp success.

T6 and T7 opened the door but later lost the grasp. Accordingly, final near-zero
aperture must not be read as proof that the door-opening task failed. Conversely,
nonzero proof aperture did not guarantee that the handle would remain held
during the longer pull. Door-state evidence is separate; see the
[retrospective](PASTEUR_INCUBATOR_DOOR_OPENING_RETROSPECTIVE.md) and the existing
endpoint evaluation report.

## Reproduction

Plot from the tracked extracted measurements (no hardware or private transcript):

```bash
python docs/plot_door_grasp_signals.py
python -m pytest -q tests/test_door_grasp_signals.py
```

Re-extract and source-hash the original local JSON files:

```bash
python docs/plot_door_grasp_signals.py --rebuild-report
```

Outputs: `door_grasp_signal_report.json`, `door_grasp_aperture_trials.{png,svg}`,
and `door_grasp_aperture_samples.{png,svg}` in `docs/assets/code_as_learning_machine/`.
No robot-control code or logging configuration is modified by this analysis.

## Did the physical trials become closer to the demonstration?

![Contact pose versus the fixed demo](assets/code_as_learning_machine/door_contact_demo_comparison.png)

The successful demonstrations' `right_gripper` channel contains only 0 and 1
in all twelve verified recordings. The collector's trigger-derived open/close
recording is consistent with this: see `teleop_collect_example.py`'s
`RecordingSample` construction. This is not the same measurement as the
continuous actual aperture read by the trial controller. Direct aperture-curve
matching would be invalid (and could incorrectly reward empty, fully closed
grasps for matching a zero-valued close command).

Instead, the additional figure compares every preserved August 8 contact pose
with the **fixed representative demo's contact pose**. E1--E3 are the three
early contact-only attempts. T1--T7 use the mapping above. The report includes
the reference and per-trial poses, numerical differences, source hashes, and
the per-demo binary-channel audit.

The robot-frame position differences initially fall from 139.7 to 103.0 to
97.9 mm, then to 1.7 mm at T1. However, later attempts are not monotonically
closer; successful T6/T7 differ by 22.0/19.4 mm, and their angular differences
are larger than T1's. No smoothing, best-so-far curve, trial omission, or fitted
registration is used to conceal this.

This metric is **absolute robot-frame EE discrepancy**, not distance to the
current handle. The user warned that the appliance had moved, and later
control incorporated live alignment. Also, the reference demo helped generate
the tested motions, so closeness to it is not held-out generalization evidence.
The supported narrative is that demonstrations corrected a large initial pose
mismatch, followed by environment-specific adjustment and verification—not
that monotonically reproducing the original absolute pose caused success.

Regeneration uses the same commands above and additionally produces
`door_contact_demo_comparison.{png,svg}`. Re-auditing raw demos requires h5py;
plotting the tracked report does not.
