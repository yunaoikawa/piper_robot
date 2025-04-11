import time
import numpy as np


class FPSCounter:
    def __init__(self):
        self.last_time = time.time()  # Mark the initial time
        self.frame_count = 0  # Counts how many frames have passed
        self.fps = 0.0  # Calculated frames per second

    def tick(self):
        # Called each frame to update the count and possibly compute fps
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_time

        # If at least one second has passed, calculate FPS and reset counters
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = current_time
            return self.fps
        return None