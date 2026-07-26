"""WetRobo — code-based autonomous wet-lab robot, MuJoCo-first.

A code-based replacement for the Pi0.5 VLA pipeline: instead of learning a neural
policy from teleop, WetRobo executes wet-lab tasks by composing parametric skills,
verifies the outcome against a CAD/MJCF-grounded scene graph (`bench_verify`), and
refines its own skill parameters across attempts ("learn by performing").

Everything runs first in a MuJoCo twin of the lab (`robot/piper-mujoco/xml/lab-scene.xml`)
because real wet-lab work is dangerous and this machine does not control the robot.

Design rule (project Principles 1/2/5): no fabricated data. Experiment numbers come from
real MuJoCo rollouts; the no-CAD baseline runs real perception on rendered RGB-D; goal
regions and geometry are derived from the loaded MJCF, never hardcoded.
"""
from wetrobo._paths import LAB_SCENE_XML, REPO_ROOT  # noqa: F401
