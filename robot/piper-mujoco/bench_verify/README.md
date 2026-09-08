# bench_verify

CAD/MJCF-grounded bench-state verification for autonomous wet-lab robotics.
A metric replacement for the camera/heuristic success signal in an
ENPIRE-style EN-module (auto reset + verify), built to drop into the Piper
robot repo on `pasteur`.

## MJCF or CAD?

Use **MJCF as the canonical container; CAD meshes are the per-object assets
inside it.** MJCF already carries identity + SE(3) pose per body, so it *is*
the scene graph, in a format the robot repo already parses (`mujoco>=3.2.7`,
existing lab scene, ConeE sim). One canonical `bench_canonical.xml` then
serves three roles: (1) the verifier's ground truth, (2) the mesh source for
FoundationPose, (3) the renderer for controlled tests. CAD and MuJoCo are not
competitors — MJCF wraps the CAD.

## Phases

**Phase 0 — does CAD help? (no robot, no GPU, runs now)**
`validate.py` Monte-Carlos four reset/verify scenarios (ok / frame_offset /
moved / missing) through the same `SceneVerifier`, comparing a vision
heuristic (degrades on transparent labware) against a CAD/model-based estimate.
Only the two estimator callables are stubs; in Phase 2 they are swapped for
real vision and real FoundationPose to regenerate the identical table on
hardware.

    python -m bench_verify.validate --n 6000
    python -m bench_verify.validate --from-mjcf bench_canonical.xml

Result (synthetic canonical; flasks/ethanol transparency-capable):

    transp%  estimator  false_success%  false_reset%  accuracy%
        0%     vision         0.75         12.05        87.20
        0%        cad         1.07          0.00        98.93
       50%     vision         0.17         52.72        47.12
       50%        cad         0.85          0.00        99.15
      100%     vision         0.02         60.57        39.42
      100%        cad         0.97          0.03        99.00

Vision accuracy collapses 87% -> 39% as transparent fraction rises, driven by
false_reset (a fine reset judged "failed" -> wasted robot time -> the exact
MRU-killing failure ENPIRE reports). CAD holds ~99% / ~0% across all fractions.
This is the paper's Table 1, and the smoking gun is already in the repo README:
"SigLIP struggles with transparent flasks."

**Phase 1 — oracle on the existing lab MuJoCo (`mujoco_oracle.py`)**
`ground_truth_state(lab.xml)` and `render_rgbd(...)` turn the lab twin into
paired (RGB-D, exact pose) data, so a real perception backend can be scored
(ADD / ADD-S) before touching the robot. `perturb_to_mjcf(...)` stages
ok/frame/moved/missing cases with known ground truth.

**Phase 2 — on hardware (`episode_hook.py`, `pose_server_peacock.py`)**
`EpisodeSceneVerifier` slots into `rollout/episode.py` `EpisodeManager`. Record3D
already provides RGB-D + intrinsics on pasteur; FoundationPose runs on peacock05
behind one extra SSH tunnel (port 15558). Reversible sub-tasks only (flask
in/out, capping, incubator load); truly irreversible steps (pipetting, mixing)
are out of scope for auto-reset.

## Files

    scene_graph.py   Item/BenchState, Kabsch transfer-diff, to/from_mjcf, to_cad
    verify.py        SceneVerifier, VerifyResult  (the EN-module verify)
    validate.py      Phase-0 experiment (does CAD help?)
    mujoco_oracle.py lab MuJoCo -> ground truth + RGB-D   (needs mujoco)
    episode_hook.py  EpisodeSceneVerifier + perception backends
    pose_server_peacock.py  FoundationPose ZMQ server stub (peacock05)

## Authoring the canonical bench by hand

    from bench_verify import to_mjcf
    from bench_verify.validate import synthetic_canonical
    to_mjcf(synthetic_canonical(), "bench_canonical.xml")  # edit poses by hand

Item bodies use the name convention `item__<id>__<label>__<kind>__<container>`;
for a pre-existing scene pass `from_mjcf(path, name_map={body: (id,label,kind,
container)})`.

## Next step on pasteur (Claude Code)

1. `python -m bench_verify.validate` to confirm the Phase-0 gap on your bench.
2. Author/adjust `bench_canonical.xml` from your real labware poses (or
   `ground_truth_state(lab.xml)` if the lab scene already has the labware).
3. Wire `EpisodeSceneVerifier` into `rollout/episode.py` per the docstring,
   starting with `MJCFOracleBackend` (no camera) to validate the loop, then
   switch to `RemotePoseBackend` once the peacock05 server is up.

Deps: numpy, scipy (core). mujoco for Phase 1. pyzmq for Phase 2.
