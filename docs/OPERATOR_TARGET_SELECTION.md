# Operator target selection

Use this workflow whenever automatic recognition has not yet established the
correct task center. It is intentionally motion-free: robot motion remains
blocked until a person taps the image and presses the explicit confirmation
button.

1. Capture a fresh synchronized frame while the robot is stopped.
2. Start the phone UI with the right image and a task-specific semantic name:

   ```bash
   PYTHONPATH=. python src/serve_target_selection_ui.py \
     --image PATH/TO/right.jpg \
     --selection artifacts/target_selection.json \
     --semantic-name microscope_stage_central_aperture \
     --port 8771
   ```

3. Open `http://TAILSCALE_IP:8771/`, tap the target center, inspect the yellow
   marker, and press **この点を中心として確定**.
4. Convert the confirmed pixel into a reusable tag-frame point. The wrist RGB
   camera has no depth, so this replaces the uncertain ray while retaining the
   prior target's camera-Z estimate:

   ```bash
   PYTHONPATH=. python src/refine_stage_aperture_from_selection.py \
     --config src/configs/pasteur_microscope_stage_aperture.json \
     --selection artifacts/target_selection.json \
     --output-config artifacts/refined_stage_aperture.json
   ```

5. Run `src/check_stage_aperture_visibility.py` on the live right image. A
   target is actionable only in state `target_visible`: tag projection alone,
   an off-frame target, blue-gripper occlusion, or an unconfirmed dark region
   all fail closed. Do not descend before this state.

The JSON stores both pixel and normalized coordinates, so image resolution is
not hard-coded. The fixed tag is only a local geometric anchor; target identity
comes from the operator selection and the aperture appearance gate.
