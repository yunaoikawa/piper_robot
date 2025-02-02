import time


class FrequencyTimer:
  def __init__(self, frequency: int):
    self.frequency = frequency  # Hz

  def __enter__(self):
    self.start = time.time()
    return self

  def __exit__(self):
    end = time.time()
    elapsed = end - self.start
    sleep_time = 1 / self.frequency - elapsed
    if sleep_time > 0:
      time.sleep(sleep_time)
    else:
      print(f"Warning: Step time {1000 * elapsed:.3f} ms in {self.__class__.__name__} control_loop")
