# ACT 2026-06-25 microscope-to-bench reproduction

This profile is isolated from the current lid agent. It reproduces the ACT
checkpoint completed at 2026-06-25 05:55 JST (Slurm job 26349).

- Dataset: `data/train/after0610/petri/microscope`
- Demonstrations: 53 episodes / 18,617 frames at 30 fps
- Training: scratch, 100,000 steps, chunk size 100, batch 16, seed 1000
- Inputs: 20D bimanual state plus head, left-wrist, and right-wrist RGB
- Checkpoint model SHA-256: `00cc3da70f04d06c9c8b67b4584607a10e0bcb6251b96304b1d2656fbc877fd8`
- Task text: `Move the petri dish to the bench.`

The historical command used `--pred_horizon 50`, but the June server did not
slice the checkpoint's 100-action output. It therefore executed the complete
100-step chunk and inferred again only after the buffer emptied. The explicit
`--legacy-full-chunk` flag recreates that behavior without changing normal ACT
serving.

Submit from the Peacock login node with explicit resources:

```bash
sbatch --partition=all --gres=gpu:1 --cpus-per-task=4 --mem=32G \
  --time=04:00:00 --job-name=act-20260625-petri-microscope \
  --output=/home/yoikawa/src/robot-vla-data/logs/act_20260625_%j.log \
  --wrap='/home/yoikawa/src/robot-vla-data/src/run_20260625_petri_microscope_act.sh'
```

The default ports are 5755/5756/5757 so this cannot collide with the current
lid-open (5555-5557) or lid-close (5655-5657) servers. Starting the server does
not command the physical robot. Do not start a controller until all three
historical camera inputs are fresh and the dish is staged at the microscope.
