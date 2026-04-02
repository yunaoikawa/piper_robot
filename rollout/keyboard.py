"""Keyboard input handling for episode control."""

import sys
import select
import tty
import termios
import threading


class KeyboardController:
    """Handles keyboard input for episode control (manual and autonomous modes)."""

    def __init__(self, stop_event: threading.Event, episode_manager, enable_recording: bool,
                 autonomous_mode: bool):
        """
        Initialize keyboard controller.

        Args:
            stop_event: Event to signal shutdown
            episode_manager: EpisodeManager instance to control
            enable_recording: Whether recording is enabled
            autonomous_mode: Whether in autonomous mode
        """
        self.stop_event = stop_event
        self.episode_manager = episode_manager
        self.enable_recording = enable_recording
        self.autonomous_mode = autonomous_mode

    def start(self):
        """Start keyboard monitoring thread."""
        self.keyboard_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self.keyboard_thread.start()

    def stop(self):
        """Stop keyboard thread."""
        self.keyboard_thread.join(timeout=2.0)

    def _keyboard_loop(self):
        """
        Thread to monitor keyboard input using stdin with non-blocking mode.

        Manual mode:
          's' - Start episode
          'e' - End episode
          'h' - Toggle arms between home/rest (only when not in episode)
          'q' - Quit

        Autonomous mode:
          's' - Resume autonomous mode (start episode + enable auto-start)
          'e' - Pause autonomous mode (end episode + disable auto-start)
          'h' - Toggle arms between home/rest (only when not in episode)
          'q' - Quit
        """
        print("\n" + "="*60)
        print("KEYBOARD CONTROLS:")
        if self.autonomous_mode:
            print("  's' - Start episode & resume autonomous mode")
            print("  'e' - End episode & pause autonomous mode")
        else:
            print("  's' - Start episode (actions applied to robot)")
            print("  'e' - End episode (stop applying actions)")
        print("  'h' - Toggle arms between home/rest (only when not in episode)")
        print("  'q' - Quit program")
        if self.enable_recording:
            print("\nRecording automatically starts/stops with episodes")
        print("="*60 + "\n")

        # Check if stdin is a TTY
        if not sys.stdin.isatty():
            print("WARNING: stdin is not a TTY, keyboard controls disabled")
            return

        # Save original terminal settings
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            # Set terminal to cbreak mode (unbuffered, no echo for our reads)
            tty.setcbreak(sys.stdin.fileno())

            while not self.stop_event.is_set():
                # Use select to check if input is available (non-blocking)
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).lower()
                    print(f"\n[Key pressed: '{key}']")

                    if key == 's':
                        if self.autonomous_mode:
                            # Autonomous mode: start episode and resume auto-start
                            if not self.episode_manager.is_active():
                                print("▶️  Starting episode...")
                                self.episode_manager.start_episode()
                            self.episode_manager.resume_autonomous()
                        else:
                            # Manual mode: just start episode
                            if not self.episode_manager.is_active():
                                print("▶️  Starting episode...")
                                self.episode_manager.start_episode()
                            else:
                                print("⚠️  Episode already active!")

                    elif key == 'e':
                        if self.autonomous_mode:
                            # Autonomous mode: pause auto-start first, then end episode
                            # (Must pause BEFORE ending to prevent race condition)
                            self.episode_manager.pause_autonomous()
                            if self.episode_manager.is_active():
                                print("⏹️  Ending episode...")
                                self.episode_manager.end_episode(reason="manual_override")
                        else:
                            # Manual mode: just end episode
                            if self.episode_manager.is_active():
                                print("⏹️  Ending episode...")
                                self.episode_manager.end_episode()
                            else:
                                print("⚠️  No active episode!")

                    elif key == 'h':
                        # Toggle between home and rest positions (only when not in episode)
                        self.episode_manager.toggle_arm_position()

                    elif key == 'q':
                        print("👋 Quitting...")
                        # End episode if active
                        if self.episode_manager.is_active():
                            self.episode_manager.end_episode()
                        self.stop_event.set()
                        break

        except Exception as e:
            print(f"Keyboard thread error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Restore original terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            print("[DEBUG] Terminal settings restored")