import time

class FrequencyTimer:
    def __init__(self, name: str, frequency: int, delay_warn_threshold: float = -1):
        self.name = name
        self.interval = int(1e9 / frequency)
        self.last_time = time.perf_counter_ns()
        self.delay_warn_threshold = int(delay_warn_threshold * 1e9)

    def __enter__(self):
        self.last_time = time.perf_counter_ns()

    def __exit__(self, *args):
        elapsed = time.perf_counter_ns() - self.last_time
        if elapsed < self.interval:
            time.sleep((self.interval - elapsed) / 1e9)
        elif self.delay_warn_threshold > 0 and elapsed > self.interval + self.delay_warn_threshold:
            print(f"{self.name} is running behind by {(self.interval - elapsed) / 1e6} ms")