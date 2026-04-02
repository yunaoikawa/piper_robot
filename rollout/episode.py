"""Episode state management and autonomous control logic."""

import time
import zmq


class EpisodeManager:
    """Manages episode state and autonomous control logic."""

    def __init__(self, recorder=None, robot_rpc=None, control_socket=None,
                 autonomous_mode=False, episode_timeout=60.0,
                 manipulability_threshold=0.05, auto_start_delay=10.0):
        """
        Initialize episode manager.

        Args:
            recorder: Optional DataRecorder instance
            robot_rpc: Robot RPC client for arm reset
            control_socket: ZMQ socket for sending control commands to HPC
            autonomous_mode: Whether to run in autonomous mode
            episode_timeout: Maximum episode duration in seconds
            manipulability_threshold: Minimum manipulability score
            auto_start_delay: Delay before auto-starting first episode
        """
        self.recorder = recorder
        self.robot_rpc = robot_rpc
        self.control_socket = control_socket
        self.autonomous_mode = autonomous_mode
        self.episode_timeout = episode_timeout
        self.manipulability_threshold = manipulability_threshold
        self.auto_start_delay = auto_start_delay

        # Episode state
        self.is_episode_active = False
        self.episode_start_time = None
        self.episode_count = 0
        self.controller_start_time = None

        # Arm position state
        self.arms_at_home = True  # Start at home position

        # Autonomous mode override
        self.autonomous_paused = False  # Manual pause of autonomous mode

    def is_active(self):
        """Check if an episode is currently active."""
        return self.is_episode_active

    def get_start_time(self):
        """Get episode start time."""
        return self.episode_start_time

    def get_count(self):
        """Get episode count."""
        return self.episode_count

    def set_controller_start_time(self):
        """Record when the controller started (for autonomous mode)."""
        self.controller_start_time = time.time()

    def pause_autonomous(self):
        """Pause autonomous episode auto-start."""
        self.autonomous_paused = True
        print("⏸️  Autonomous mode PAUSED - Press 's' to resume")

    def resume_autonomous(self):
        """Resume autonomous episode auto-start."""
        # Only allow resuming autonomous mode from home position
        if not self.arms_at_home:
            print("⚠️  Cannot resume autonomous mode - arms must be at home position!")
            print("   Press 'h' to toggle arms to home position first.")
            return

        self.autonomous_paused = False
        print("▶️  Autonomous mode RESUMED - Episodes will auto-start")

    def toggle_arm_position(self):
        """Toggle arms between home and rest positions (only when not in episode)."""
        if self.is_episode_active:
            print("⚠️  Cannot toggle arm position during episode!")
            return False

        if not self.robot_rpc:
            print("⚠️  Robot RPC not available!")
            return False

        if self.arms_at_home:
            print("🔄 Moving arms to rest position...")
            self.robot_rpc.rest_left_arm()
            self.robot_rpc.rest_right_arm()
            self.arms_at_home = False
            print("✓ Arms at rest position")
        else:
            print("🔄 Moving arms to home position...")
            self.robot_rpc.home_left_arm()
            self.robot_rpc.home_right_arm()
            self.arms_at_home = True
            print("✓ Arms at home position")

        return True

    def start_episode(self):
        """Start a new episode (enables action execution and recording)."""
        if self.is_episode_active:
            print("⚠️  Episode already active!")
            return

        # Only allow starting episode from home position
        if not self.arms_at_home:
            print("⚠️  Cannot start episode - arms must be at home position!")
            print("   Press 'h' to toggle arms to home position first.")
            return

        self.clear_action_queue()

        self.is_episode_active = True
        self.episode_start_time = time.time()

        # Automatically start recording if enabled
        if self.recorder:
            self.recorder.start_episode()

        print(f"▶️  Episode {self.episode_count} started - Actions now being applied to robot")

    def clear_action_queue(self):
        """Send command to HPC to clear the action queue."""
        if not self.control_socket:
            return

        try:
            command = {'command': 'clear_queue'}
            self.control_socket.send_pyobj(command)
            response = self.control_socket.recv_pyobj()

            if response.get('status') == 'ok':
                print("✓ Action queue cleared on HPC")
            else:
                print(f"⚠️  Failed to clear action queue: {response.get('message', 'Unknown error')}")
        except zmq.error.Again:
            print("⚠️  Timeout: Could not clear action queue (HPC not responding)")
        except Exception as e:
            print(f"⚠️  Error clearing action queue: {e}")

    def end_episode(self, reason="manual"):
        """End current episode (disables action execution and saves recording)."""
        if not self.is_episode_active:
            return

        self.is_episode_active = False
        episode_duration = time.time() - self.episode_start_time if self.episode_start_time else 0

        # Clear action queue on HPC server
        # self.clear_action_queue()

        # Automatically end recording if enabled
        if self.recorder and self.recorder.is_recording:
            self.recorder.end_episode()

        print(f"⏹️  Episode ended ({reason}) - Duration: {episode_duration:.1f}s")

        # Reset arm positions after episode ends
        if self.robot_rpc:
            print("🏠 Resetting arm positions to home...")
            try:
                self.robot_rpc.home_left_arm()
                self.robot_rpc.home_right_arm()
                time.sleep(2.0)  # Wait for arms to reach home
                self.arms_at_home = True  # Update state
                print("✓ Arms reset complete")
            except Exception as e:
                print(f"⚠️  Warning: Could not reset arms: {e}")

        self.episode_start_time = None
        self.episode_count += 1

    def check_autonomous_conditions(self, manipulability_calculator, iteration):
        """
        Check autonomous mode conditions and manage episodes accordingly.

        Args:
            manipulability_calculator: ManipulabilityCalculator instance
            iteration: Current control loop iteration

        Returns:
            True if episode was modified (started/ended), False otherwise
        """
        if not self.autonomous_mode:
            return False

        # Don't auto-start if autonomous mode is manually paused
        if self.autonomous_paused:
            return False

        current_time = time.time()

        # Auto-start first episode after delay
        if not self.is_episode_active and self.controller_start_time is not None:
            time_since_start = current_time - self.controller_start_time
            if time_since_start >= self.auto_start_delay:
                print(f"🤖 Auto-starting episode (delay: {self.auto_start_delay}s elapsed)")
                self.start_episode()
                return True

        # Check for episode timeout or low manipulability
        if self.is_episode_active:
            episode_duration = current_time - self.episode_start_time

            # Check timeout
            if episode_duration >= self.episode_timeout:
                print(f"⏱️  Episode timeout ({self.episode_timeout}s)")
                self.end_episode(reason="timeout")
                # Auto-restart after brief pause (if not paused)
                if not self.autonomous_paused:
                    time.sleep(2.0)
                    print("🤖 Auto-restarting new episode")
                    self.start_episode()
                return True

            # Check manipulability
            try:
                manipulability = manipulability_calculator.calculate(arm='right')

                if manipulability < self.manipulability_threshold:
                    print(f"🔧 Low manipulability detected ({manipulability:.3f} < {self.manipulability_threshold})")
                    self.end_episode(reason="low_manipulability")
                    # Auto-restart after brief pause (if not paused)
                    if not self.autonomous_paused:
                        time.sleep(2.0)
                        print("🤖 Auto-restarting new episode")
                        self.start_episode()
                    return True
            except Exception as e:
                # Don't crash on manipulability calculation errors
                if iteration % 100 == 0:
                    print(f"Warning: Could not calculate manipulability: {e}")

        return False