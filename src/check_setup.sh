#!/bin/bash
# Verify the pasteur checkout has everything the replay + bias + safety workflow
# needs, and that controller.py is the MERGED version (camera fixes + safety),
# not one of the half-states from the 2026-07-21 file-shuffling.
#
#   bash src/check_setup.sh
#
# Read-only: it inspects files and tries imports, it does not move the robot.

set -u
cd "$(dirname "$0")/.." || exit 1
fail=0

ok()   { echo "  OK    $1"; }
bad()  { echo "  FAIL  $1"; fail=1; }

echo "=== required files present ==="
for f in \
    rollout/safety.py \
    rollout/torque_safety.py \
    rollout/lid_vision.py \
    rollout/controller.py \
    robot/cone_e.py \
    robot/camera_id.py \
    robot/camera_map.json \
    src/set_bias.py \
    src/replay_checkpoint.py \
    src/configs/safety.json \
    src/configs/pasteur_lid_vision.json \
    cloud_inference_control_collect_v2.py \
    replay_demo.py ; do
    [[ -f "$f" ]] && ok "$f" || bad "$f MISSING"
done

echo "=== controller.py is the MERGED version ==="
# Both markers must be present: camera fix (pasteur) AND safety layer (peacock).
grep -q "load_camera_map" rollout/controller.py && ok "camera fix present" \
    || bad "camera fix MISSING -- this is my non-merged version, cameras will break"
grep -q "from .safety import SafetyLayer" rollout/controller.py && ok "safety layer present" \
    || bad "safety layer MISSING -- this is the old pasteur version, no bias/safety"
grep -q "_rotate_and_resize" rollout/controller.py && ok "480x640 resize present" \
    || bad "resize MISSING"
grep -q "def set_bias" rollout/controller.py && ok "set_bias present" || bad "set_bias MISSING"

echo "=== cone_e.py clamp state + bounds ==="
if grep -qE "^\s*ee_target = clamp_ee_target" robot/cone_e.py; then
    ok "workspace clamp ENABLED"
else
    bad "workspace clamp is commented out -- bias will not be bounded by the box"
fi
echo "  bounds: $(grep -E 'WORKSPACE_(MIN|MAX)' robot/cone_e.py | tr -s ' ')"

echo "=== camera_map.json contents ==="
echo "  $(cat robot/camera_map.json 2>/dev/null || echo MISSING)"

echo "=== imports (class defs only; does not touch hardware) ==="
python -c "from rollout.controller import PolicyController; from rollout.safety import SafetyLayer; print('  OK    controller + safety import')" \
    || bad "import failed (see traceback above)"

echo "=== safety layer unit tests ==="
if [[ -f test/test_safety_layer.py ]]; then
    python test/test_safety_layer.py >/dev/null 2>&1 && ok "test_safety_layer passed" \
        || bad "test_safety_layer FAILED -- run 'python test/test_safety_layer.py' to see why"
else
    echo "  (test/test_safety_layer.py not copied -- optional)"
fi

echo "=== replay preflight unit tests ==="
if [[ -f tests/test_replay_demo.py ]]; then
    python tests/test_replay_demo.py >/dev/null 2>&1 && ok "test_replay_demo passed" \
        || bad "test_replay_demo FAILED -- run 'python tests/test_replay_demo.py' to see why"
else
    bad "tests/test_replay_demo.py MISSING"
fi

echo "=== vision + torque watchdog unit tests ==="
for test_file in tests/test_lid_vision.py tests/test_torque_safety.py; do
    if python "$test_file" >/dev/null 2>&1; then
        ok "$test_file passed"
    else
        bad "$test_file FAILED -- run 'python $test_file' to see why"
    fi
done

echo
if [[ "$fail" -eq 0 ]]; then
    echo "ALL CHECKS PASSED -- ready for the regression check (docs/PASTEUR_REPLAY_WORKFLOW.md)."
else
    echo "SOME CHECKS FAILED -- fix the FAIL lines before running the robot."
    exit 1
fi
