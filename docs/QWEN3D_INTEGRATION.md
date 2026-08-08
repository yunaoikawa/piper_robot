# Qwen-3D integration for semantic lab scenes

Qwen-3D is a language-conditioned 3D grounding, instance-segmentation, and
question-answering model. It is not a replacement for Record3D registration,
TSDF fusion, object completion, ESDF, or MuJoCo collision geometry. In this
repository its intended position is:

```text
posed RGB-D -> metric reconstruction -> Qwen-3D language grounding
                                     -> semantic object completion
                                     -> ESDF / MuJoCo / planning gates
```

SAM remains useful as a cheap bootstrap and regression source. Qwen-3D can
resolve relational requests such as “the incubator on the rear platform” or
“the bottle between the platforms” and return a dense 3D mask. A Qwen-3D mask
does not authorize motion by itself; metric depth, support, calibration,
collision, and trajectory gates remain mandatory.

## Input adapter and phone preview

The observation-only adapter exports the fused metric point cloud plus five
posed, RGB/depth-aligned frames. It also writes an interactive phone view.

```bash
PY=/home/admin/miniforge3/envs/robot-test/bin/python
$PY src/export_qwen3d_scene.py \
  --multiview-report \
    data/reconstructions/pasteur/record3d_20260730_215729_v3/multiview_report.json \
  --output-dir data/runs/pasteur/qwen3d_input
```

Artifacts:

- `manifest.json`: input provenance, coordinate frame, queries, readiness
- `qwen3d_input_points.npz`: XYZ metres, RGB, SAM bootstrap labels
- `posed_rgbd/*/color.png`: RGB resized to the organized depth grid
- `posed_rgbd/*/depth.png`: uint16 millimetre depth
- `posed_rgbd/*/{intrinsic,pose}.txt`: per-view camera geometry
- `qwen3d_upstream_rgbd/pasteur_scene/*`: 512x512 RGB-D, pose, and adjusted
  intrinsics for the reviewed upstream AI2-THOR-style loader
- `index.html`: RGB and per-object 3D visibility controls

The upstream copy uses nearest-neighbour depth resampling and scales `fx`,
`fy`, `cx`, and `cy` independently. This preserves backprojected camera rays
even though the source Record3D image is not square.

The viewer title and manifest explicitly distinguish existing SAM bootstrap
labels from Qwen-3D predictions.

## Runtime requirements and upstream boundary

The reviewed upstream revision is
`7ef6d01e495290639884878a06e43ba2905e0ef5`. The 3B Qwen-3D checkpoint is
approximately 7.9 GB, in addition to the Qwen2.5-VL-3B backbone and compiled
CUDA dependencies (`torch-scatter`, Detectron2, PyTorch3D, and custom point
operators). Run inference on a CUDA GPU node, not the Pasteur controller.

The reviewed upstream repository provides benchmark training/evaluation entry
points rather than a finished custom-scene CLI. Keep its environment and
source outside `piper_robot`; adapt the manifest at the boundary. No upstream
source is vendored here. The reviewed revision does not contain a top-level
license file, so copying its implementation into this repository is not
assumed to be permitted.

## Output contract

A future inference wrapper must return a separate artifact containing:

- upstream revision and checkpoint hashes;
- natural-language query;
- point order/hash matching `qwen3d_input_points.npz`;
- per-point mask probability and selected boolean mask;
- confidence, inference time, and GPU identity;
- explicit `motion_authority=false`.

The semantic scene pipeline may compare this result against SAM and depth,
but must preserve disagreements as uncertainty rather than replacing one mask
silently.

After inference, render the prediction contract without mixing it into the
bootstrap SAM viewer:

```bash
$PY src/render_qwen3d_predictions.py \
  --predictions data/runs/pasteur/qwen3d_predictions/predictions.json \
  --output data/runs/pasteur/qwen3d_predictions/index.html
```
