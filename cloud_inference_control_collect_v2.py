"""Robot policy controller with episode control and data collection (Refactored)."""

import atexit
import argparse

from rollout import PolicyController


def main():
    parser = argparse.ArgumentParser(
        description='Robot policy controller with recording and autonomous mode'
    )
    parser.add_argument('--host', type=str, default='192.168.1.50',
                        help='HPC server IP address')
    parser.add_argument('--obs-port', type=int, default=5555,
                        help='Port for publishing observations')
    parser.add_argument('--action-port', type=int, default=5556,
                        help='Port for requesting actions')
    parser.add_argument('--rate', type=int, default=8,
                        help='Control rate in Hz')
    parser.add_argument('--record', action='store_true',
                        help='Enable recording of observations')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save recordings')
    parser.add_argument('--autonomous', action='store_true',
                        help='Enable autonomous mode (auto start/end episodes)')
    parser.add_argument('--episode-timeout', type=float, default=30.0,
                        help='Episode timeout in seconds (default: 30)')
    parser.add_argument('--manipulability-threshold', type=float, default=0.005,
                        help='Minimum manipulability score (default: 0.005)')
    parser.add_argument('--task', type=str, default='put the flask in the incubator',
                        help='Task description for the policy')
    args = parser.parse_args()

    # Create controller
    controller = PolicyController(
        hpc_host=args.host,
        obs_port=args.obs_port,
        action_port=args.action_port,
        enable_recording=args.record,
        save_dir=args.save_dir,
        autonomous_mode=args.autonomous,
        episode_timeout=args.episode_timeout,
        manipulability_threshold=args.manipulability_threshold,
        task=args.task,
    )
    atexit.register(controller.stop)

    # Print configuration
    print("\n" + "="*60)
    print("POLICY CONTROLLER WITH EPISODE CONTROL")
    print("="*60)

    if args.autonomous:
        print("\n🤖 AUTONOMOUS MODE ENABLED")
        print(f"  Auto-start delay: 10.0s")
        print(f"  Episode timeout: {args.episode_timeout}s")
        print(f"  Manipulability threshold: {args.manipulability_threshold}")
        print("  Episodes will auto-start and auto-end")
    else:
        print("\n⌨️  MANUAL MODE")
        print("  Keyboard Controls:")
        print("    's' - Start episode (begin applying actions)")
        print("    'e' - End episode (stop applying actions)")
        print("    'q' - Quit program")

    if args.record:
        print("\n🔴 RECORDING ENABLED")
        print(f"  Data saved to: {controller.recorder.save_dir}")
        print("  Recording automatically syncs with episodes")
    else:
        print("\n📷 Recording disabled (use --record to enable)")

    print(f"\n📝 Task: '{args.task}'")
    print("\n💡 Live camera feed will display in a window")
    print("="*60 + "\n")

    # Run control loop
    try:
        controller.control_loop(control_rate=args.rate)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        controller.stop()


if __name__ == "__main__":
    main()