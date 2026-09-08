# Internal evidence map (not an anonymous submission artifact)

The PDFs do not link to an identifying repository. This map is for author
verification and reproducible analysis, not a claim that every raw record is
included in the anonymous submission.

| Claim / figure | Primary or audited source | Interpretation |
|---|---|---|
| 12 successful opening episodes from 57 candidates | `data/reference/pasteur/incubator/compiled_door_open_v1.json`; `../assets/code_as_learning_machine/door_grasp_signal_report.json:demo_comparison` | Full episodes, selected using final head images; not 12 state samples |
| Demo was introduced by operator | Access-controlled Aug 6 session archive, Aug 7 02:22:32 UTC user message, previously audited | Existing archive offered by user, not newly requested teleoperation |
| Relative door trajectory and checks | `f2149bb:src/run_incubator_door_demo.py`; `f2149bb:rollout/teleop_trajectory_stream.py` | Inspect historical source, not later unrelated edits |
| Door orchestration | `f2149bb:src/run_incubator_door_autonomy.py`; endpoint refinement `283b913` | Executable control and object-state confirmation |
| T1–T7 aperture observations | `../assets/code_as_learning_machine/door_grasp_signal_report.json` | Preserve missing endpoint values; normalized aperture is not pressure |
| T7-reference orientation | `../assets/code_as_learning_machine/door_contact_t7_comparison_orientation.json` | Recomputed full SO(3) contact-pose difference; T7 zero is definitional |
| Closed/open photographs | `door_grasp_overlay_T{1,2,6,7}_head.jpg` and frozen `door_configuration_curve_report.json` in existing assets | Later endpoint images, not synchronized slip timing |
| Incomplete full-pull trace | `../assets/code_as_learning_machine/door_pull_trace_audit.json`; `../DOOR_GRASP_SIGNAL_AUDIT.md` | RAM checkpoints lost on exceptions; no synthetic time curve |
| Cap local servo / target | `fc831c0:rollout/media_cap_target.py`; `fc831c0:src/run_culture_media_cap_grasp.py` | Fixed-head 2x3 Jacobian and pseudoinverse; no cap task demo |
| Cap verification and scoped route | `1f07761:rollout/cylindrical_cap_transfer.py` | Source/support/retention checks; exact-route exception |
| Cap measured rise and transport | Three archived Aug 6 manifests listed in `evidence.json` | Differences of `robot_state.before.right_ee_pose.translation_xyz_m`; aperture from same snapshots |
| Cap RGB sequence | `../assets/code_as_learning_machine/cap_verified_transfer_rgb.jpg` | Full frames plus labeled crops of the same three captures |
| SAM absent from promoted paths | Historical source inventory in `../CODE_AS_A_LEARNING_MACHINE_MANUSCRIPT.md`, Appendix F | Constants imported from SAM-named module do not imply model execution |
| Bio scope | `../PASTEUR_CULTURE_MEDIA_CAP_TRANSFER_RETROSPECTIVE.md` | Cap removal/transport, not threaded unscrewing, placement, sterility, or assay outcome |

`extract_evidence.py` records hashes of the relevant historical source objects.
`check_papers.py` verifies tracked numerical values, input hashes, PDF pages,
fonts, anonymous text and unresolved references. It does not claim to test the
robot or to reproduce a physical success from the PDF.

## Additional author checks before submission

- Historical Codex backend/model revisions and exact intervention budgets have
  not been pinned here. The drafts identify Codex but do not invent a model ID.
- Development evaluator changes and later retrospective analysis are different
  from a pre-registered fixed evaluation. The papers say so explicitly.
- The aperture 0.296 in rounded T7 tables is the saved proof median; the
  approximately 0.2957 stationary recheck is a separate observation.
- Hardware tuple precision is not calibrated metrology accuracy. Round physical
  effects to 9.17 mm and 107.42 mm rather than advertising six-digit accuracy.
- Lab/manufacturer objects in images are retained as physical evidence; no
  author face, account identifier, or repository URL is shown in the figures.
- The cap case predates the door case in the archived sequence. Neither draft
  claims that the completed door controller was transferred to the cap.

## Literature and formatting sources checked 2026-09-07

- ENPIRE: https://arxiv.org/html/2606.19980v1 (Sections 2.1–2.2, 3.5).
- Code as Policies: https://arxiv.org/abs/2209.07753.
- VoxPoser: https://arxiv.org/abs/2307.05973.
- OpenVLA: https://arxiv.org/abs/2406.09246.
- pi0.5: https://arxiv.org/abs/2504.16054.
- MuJoCo: https://mujoco.org/ and DOI 10.1109/IROS.2012.6386109.
- Laboratory robotics: https://www.nature.com/articles/s41586-020-2442-2.
- SAM and ReAct bibliographic entries retained from the source manuscript.
- ICRA initial submission rules and official template URLs are in README.md.
