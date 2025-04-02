import os
from typing import cast
import time
from queue import Queue
import threading

os.environ["CTR_TARGET"] = "Hardware"
import phoenix6.unmanaged
from phoenix6 import configs, controls, hardware

from robot.network.timer import FrequencyTimer
from robot.controller.constants import CONTROL_FREQ, POLICY_CONTROL_PERIOD_NS, POLICY_CONTROL_FREQ

class Lift:
    def __init__(self):
        self.lift_motor = hardware.TalonFX(9, canbus="Drivetrain")

        time.sleep(0.2)

        assert self.lift_motor.get_is_pro_licensed()

        self.position_signal = self.lift_motor.get_position()
        self.velocity_signal = self.lift_motor.get_velocity()
        self.status_signals = cast(list[phoenix6.BaseStatusSignal], [self.position_signal, self.velocity_signal])
        self.position_request = controls.MotionMagicTorqueCurrentFOC(0)
        self.neutral_request = controls.NeutralOut()

        self.lift_motor_cfg = configs.TalonFXConfiguration()
        self.lift_motor_cfg.slot0.k_p = 50
        self.lift_motor_cfg.slot0.k_i = 0.0
        self.lift_motor_cfg.slot0.k_d = 0.0
        self.lift_motor_cfg.torque_current.peak_forward_torque_current = 100
        self.lift_motor_cfg.torque_current.peak_reverse_torque_current = -100
        self.lift_motor_cfg.audio.beep_on_boot = False

        self.lift_motor_cfg.motion_magic.motion_magic_cruise_velocity = 15.0 # [rev/s]
        self.lift_motor_cfg.motion_magic.motion_magic_acceleration = 25.0 # [rev/s^2]

        self.min_pos, self.max_pos = 0.0, 0.39 # [m]

        phoenix6.BaseStatusSignal.set_update_frequency_for_all(CONTROL_FREQ, self.status_signals)

        status = self.lift_motor.configurator.apply(self.lift_motor_cfg)
        if not status.is_ok():
            raise Exception(f"Failed to apply TalonFX configuration: {status}")

        self._command_queue: Queue[float] = Queue(1)
        self._disable_motors = True
        self.last_command_time = time.perf_counter_ns()

        self.control_loop_thread: threading.Thread | None = threading.Thread(target=self.control_loop)
        self.control_loop_running = False

        # self.lift_motor.set_position(0)

    def get_position(self) -> float:
        # zero position - fully retracted - 0
        # lead screw pitch = 6mm per rev
        # motor pulley teeth = 16
        # shaft pulley teeth = 24
        # 16/24 * 6 = 4mm per rev
        return -self.position_signal.value * 0.004 # [m]

    def get_velocity(self) -> float:
        return -self.velocity_signal.value * 0.004 # [m/s]

    def set_target_position(self, position: float) -> None:
        assert self.min_pos <= position <= self.max_pos
        self._command_queue.put(position)

    def update_state(self):
        phoenix6.BaseStatusSignal.refresh_all(self.status_signals)

    def control_loop(self):
        disable_motors = True
        last_command_time = time.perf_counter_ns()
        timer = FrequencyTimer(name="lift_control_loop", frequency=CONTROL_FREQ)

        while self.control_loop_running:
            with timer:
                self.update_state()

                if not self._command_queue.empty():
                    command = self._command_queue.get()
                    last_command_time = time.perf_counter_ns()

                    disable_motors = abs(command - self.get_position()) < 0.003 # apply deadband - 3mm

                if (time.perf_counter_ns() - last_command_time) > 2.5 * POLICY_CONTROL_PERIOD_NS:
                    disable_motors = True

                if disable_motors:
                    self.lift_motor.set_control(self.neutral_request)
                else:
                    phoenix6.unmanaged.feed_enable(0.1)
                    self.lift_motor.set_control(self.position_request.with_position(-command / 0.004))

    def start_control(self):
        if self.control_loop_thread is None:
            print("To initiate a new control loop, create a new instance of Lift first")
            return
        self.control_loop_running = True
        self.control_loop_thread.start()

    def stop_control(self):
        if self.control_loop_thread is None:
            print("No control loop to stop")
            return
        self.control_loop_running = False
        self.control_loop_thread.join()
        self.control_loop_thread = None


def main():
    lift = Lift()
    lift.start_control()

    print("Lift initialized")
    print(f"Lift position: {lift.get_position()} m")

    timer = FrequencyTimer(name="lift_control_loop", frequency=POLICY_CONTROL_FREQ)
    try:
        target = float(input("Enter target position: "))
    except ValueError:
        print("Invalid input")
        return

    try:
        while True:
            with timer:
                lift.set_target_position(target)
    except KeyboardInterrupt:
        lift.stop_control()


if __name__ == "__main__":
    main()
