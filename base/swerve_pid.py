import numpy as np

class SteerPIDController:
    def __init__(self, dt, kp, ki=0.0, kd=0.0, num_swerves=4, max_output=5.0, min_output=-5.0, deadband=0.2):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = np.zeros(num_swerves)
        self.integral = np.zeros(num_swerves)

        self.dt = dt
        self.max_output = max_output
        self.min_output = min_output
        self.deadband = deadband

    def update(self, setpoint:np.ndarray, actual: np.ndarray):
        error = setpoint - actual
        # wrap error
        error = (error + np.pi) % (2 * np.pi) - np.pi

        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        output = np.clip(output, self.min_output, self.max_output)

        # deadband
        for i in range(len(output)):
            if abs(output[i]) < self.deadband:
                output[i] = 0

        return output, error



