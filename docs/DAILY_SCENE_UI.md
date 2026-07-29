# Daily bench confirmation

Autonomous motion binds every plan to one operator-confirmed bench revision.
The confirmation service is deliberately independent of Codex.

Start the phone UI on the robot computer:

```bash
conda run -n robot-test python src/daily_scene_ui.py \
  --scene runs/pasteur_daily_scene.json --host 0.0.0.0 --port 8765
```

Open `http://<pasteur-tailscale-ip>:8765/` on the phone. On the first autonomous
preflight of each local day, the runner saves the head SAM image and expected
inventory, then exits without moving while the scene is unconfirmed. Confirm
or mark every proposed item absent and press **この状態で正しい**.

Press **机上を変更した** whenever an object is moved, added, or removed. This
increments the scene revision immediately. A plan bound to the previous
revision holds before its next 0.5-second chunk and cannot resume until a new
scan is confirmed. A large SAM/RGB-D target shift performs the same
invalidation automatically.

The JSON API is also the integration point for chat or voice:

- `GET /api/scene`
- `POST /api/objects` with `revision` and the complete edited `objects` list
- `POST /api/confirm` with `revision` and `operator`
- `POST /api/changed` with `reason`

Unknown regions cannot be confirmed. User confirmation resolves identity and
inventory; it never converts unobserved RGB-D space into free space.

Replay a saved planned/recorded joint sequence before hardware execution:

```bash
conda run -n robot-test python src/replay_scene_collision.py \
  --model robot/cone-e-description/robot-welded-base-and-lift.mjcf \
  --q-waypoints /tmp/run_state.json --esdf /tmp/live_esdf_0000.npz \
  --daily-scene runs/pasteur_daily_scene.json \
  --output /tmp/collision_replay.json
```
