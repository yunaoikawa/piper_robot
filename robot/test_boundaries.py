#!/usr/bin/env python3
"""
Workspace boundary tester for ConeE.

Modes:
  1. EXPLORE  — move arms freely via teleop and watch live EE positions.
                Hit ENTER to snapshot the current position as a candidate boundary.
  2. TEST     — enter min/max values manually and verify the clamp is working
                by pushing the arm to the limits.

Usage:
    python test_boundaries.py            # connects to localhost:8081
    python test_boundaries.py --host 192.168.1.10 --port 8081
"""

import argparse
import threading
import time

import numpy as np

from robot.rpc import RPCClient

# ──────────────────────────────────────────────────────────────────────────────
POLL_HZ = 20   # how often to refresh the live display
# ──────────────────────────────────────────────────────────────────────────────


def fmt(arr, decimals=3):
    return "[" + "  ".join(f"{v:+.{decimals}f}" for v in arr) + "]"


class BoundaryTester:
    def __init__(self, host, port):
        print(f"Connecting to RPC server at {host}:{port} …")
        self.robot = RPCClient(host, port)
        self.robot.init()
        print("Connected.\n")

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._left_pos = np.zeros(3)
        self._right_pos = np.zeros(3)

        # running min/max seen during explore
        self._obs_min = np.full(3, np.inf)
        self._obs_max = np.full(3, -np.inf)

        self._snapshots = []   # list of (label, left_pos, right_pos)

        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    # ── background polling ──────────────────────────────────────────────────
    def _poll_loop(self):
        interval = 1.0 / POLL_HZ
        while not self._stop.is_set():
            try:
                lp = self.robot.get_left_ee_pose()
                rp = self.robot.get_right_ee_pose()
                if lp is not None and rp is not None:
                    lpos = lp.translation()
                    rpos = rp.translation()
                    with self._lock:
                        self._left_pos = lpos
                        self._right_pos = rpos
                        for pos in (lpos, rpos):
                            self._obs_min = np.minimum(self._obs_min, pos)
                            self._obs_max = np.maximum(self._obs_max, pos)
            except Exception as e:
                print(f"\n[poll] error: {e}")
            time.sleep(interval)

    def _get_poses(self):
        with self._lock:
            return np.array(self._left_pos), np.array(self._right_pos)

    def _get_observed_bounds(self):
        with self._lock:
            return np.array(self._obs_min), np.array(self._obs_max)

    # ── live display ─────────────────────────────────────────────────────────
    def _live_display(self, duration_s=None, header=""):
        """Print live EE positions. Press ENTER to snapshot, q+ENTER to quit."""
        import sys, select

        print(header)
        print("  Move the arms freely. Press ENTER to snapshot, type 'q'+ENTER to stop.\n")
        print(f"  {'AXIS':<6}  {'X':>9}  {'Y':>9}  {'Z':>9}")
        print("  " + "-" * 38)

        start = time.time()
        snapshot_count = 0

        while True:
            lpos, rpos = self._get_poses()
            obs_min, obs_max = self._get_observed_bounds()

            lines = [
                f"\r\033[6A",   # move cursor up 6 lines to overwrite
                f"  LEFT   {fmt(lpos)}",
                f"  RIGHT  {fmt(rpos)}",
                f"  obs_min{fmt(obs_min)}",
                f"  obs_max{fmt(obs_max)}",
                f"  snapshots: {snapshot_count}   ",
                "",
            ]
            print("\n".join(lines), end="", flush=True)

            # non-blocking stdin check
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                line = sys.stdin.readline().strip().lower()
                if line == "q":
                    break
                else:
                    label = f"snap_{snapshot_count}"
                    self._snapshots.append((label, np.array(lpos), np.array(rpos)))
                    snapshot_count += 1
                    print(f"\n  [snapshot {snapshot_count-1}] L={fmt(lpos)}  R={fmt(rpos)}")

            if duration_s is not None and (time.time() - start) > duration_s:
                break

        print()

    # ── MODE 1: explore ──────────────────────────────────────────────────────
    def run_explore(self):
        print("=" * 60)
        print("  MODE: EXPLORE")
        print("=" * 60)
        print("  Home the robot, then move the arms to every corner of the")
        print("  workspace you want to allow. The observed min/max will be")
        print("  tracked automatically.\n")
        input("  Press ENTER when ready to start …\n")

        self._live_display(header="\n  Live EE positions (metres)")

        obs_min, obs_max = self._get_observed_bounds()

        # add a small safety margin
        margin = 0.02
        suggested_min = obs_min - margin
        suggested_max = obs_max + margin

        print("\n" + "=" * 60)
        print("  RESULTS")
        print("=" * 60)
        print(f"  Observed min  : {fmt(obs_min)}")
        print(f"  Observed max  : {fmt(obs_max)}")
        print(f"  Suggested min : {fmt(suggested_min)}  (with {margin}m margin)")
        print(f"  Suggested max : {fmt(suggested_max)}  (with {margin}m margin)")

        if self._snapshots:
            print(f"\n  Snapshots taken ({len(self._snapshots)}):")
            for label, lpos, rpos in self._snapshots:
                print(f"    {label}:  L={fmt(lpos)}  R={fmt(rpos)}")

        print("\n  ── Copy into cone_e.py ─────────────────────────────────")
        print(f"  WORKSPACE_MIN = np.array([{suggested_min[0]:.3f}, {suggested_min[1]:.3f}, {suggested_min[2]:.3f}])")
        print(f"  WORKSPACE_MAX = np.array([{suggested_max[0]:.3f}, {suggested_max[1]:.3f}, {suggested_max[2]:.3f}])")
        print("  " + "─" * 55 + "\n")

    # ── MODE 2: test clamp ───────────────────────────────────────────────────
    def run_test(self):
        print("=" * 60)
        print("  MODE: TEST CLAMP")
        print("=" * 60)
        print("  Enter boundaries to test, then move the arm past them.")
        print("  The live display will show when clamping kicks in.\n")

        def ask_vec(label, default):
            raw = input(f"  {label} [{' '.join(str(v) for v in default)}]: ").strip()
            if not raw:
                return np.array(default, dtype=float)
            try:
                vals = [float(v) for v in raw.split()]
                if len(vals) == 3:
                    return np.array(vals)
            except ValueError:
                pass
            print("  Invalid input, using default.")
            return np.array(default, dtype=float)

        # get current poses as sensible defaults
        lpos, rpos = self._get_poses()
        default_min = np.round(np.minimum(lpos, rpos) - 0.1, 3)
        default_max = np.round(np.maximum(lpos, rpos) + 0.1, 3)

        ws_min = ask_vec("WORKSPACE_MIN (x y z)", default_min)
        ws_max = ask_vec("WORKSPACE_MAX (x y z)", default_max)

        print(f"\n  Testing with:")
        print(f"    WORKSPACE_MIN = {fmt(ws_min)}")
        print(f"    WORKSPACE_MAX = {fmt(ws_max)}")
        print("\n  Move the arms toward and past the boundaries.")
        print("  You should feel/see the arm stop at the limit.\n")
        print("  CLAMP column shows how far outside the boundary each axis is.\n")

        import sys, select

        print(f"  {'ARM':<6} {'POS':>34}   {'CLAMPED?'}")
        print("  " + "-" * 55)

        def clamp_info(pos):
            clamped = np.clip(pos, ws_min, ws_max)
            diff = pos - clamped
            hit = any(abs(d) > 1e-4 for d in diff)
            tag = f"  ← CLAMPED {fmt(diff)}" if hit else ""
            return fmt(clamped), tag

        while True:
            lpos, rpos = self._get_poses()
            l_clamped, l_tag = clamp_info(lpos)
            r_clamped, r_tag = clamp_info(rpos)

            lines = [
                f"\r\033[3A",
                f"  LEFT   raw={fmt(lpos)}  {l_tag}   ",
                f"  RIGHT  raw={fmt(rpos)}  {r_tag}   ",
                "",
            ]
            print("\n".join(lines), end="", flush=True)

            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                line = sys.stdin.readline().strip().lower()
                if line == "q":
                    break

        print("\n  Done.\n")

    # ── cleanup ───────────────────────────────────────────────────────────────
    def stop(self):
        self._stop.set()


# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Interactive workspace boundary tester for ConeE.")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument(
        "--mode",
        choices=["explore", "test", "both"],
        default="both",
        help="explore=find bounds, test=verify clamp, both=run both (default)",
    )
    args = ap.parse_args()

    tester = BoundaryTester(args.host, args.port)

    try:
        if args.mode in ("explore", "both"):
            tester.run_explore()
        if args.mode in ("test", "both"):
            again = input("  Run clamp test now? [Y/n]: ").strip().lower()
            if again != "n":
                tester.run_test()
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        tester.stop()


if __name__ == "__main__":
    main()