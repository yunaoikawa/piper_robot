import time


class RateKeeper:
  def __init__(self, name: str, rate: float, print_delay_threshold: float= 0):
    self.name = name
    self.interval = 1 / rate
    self.last_monitor_time = time.monotonic()
    self.next_frame_time = self.last_monitor_time + self.interval
    self._remaining = 0
    self.print_delay_threshold = max(0.0, print_delay_threshold)
    self.frame = 0

  def keep_time(self) -> bool:
    lagged = self.monitor_time()
    if self._remaining > 0: time.sleep(self._remaining)
    return lagged

  def monitor_time(self) -> bool:
    self.frame += 1
    self.last_monitor_time = time.monotonic()
    self._remaining = self.next_frame_time - self.last_monitor_time
    lagged = self._remaining < 0
    if lagged:
      if (self.print_delay_threshold > 0 and self._remaining < -self.print_delay_threshold):
        print(f"{self.name} lagging by {-self._remaining * 1000:.2f}ms")
      self.next_frame_time = self.last_monitor_time + self.interval
    else:
      self.next_frame_time += self.interval
    return lagged

class AsyncRateKeeper:
  def __init__(self, name: str, rate: float, print_delay_threshold: float= 0):
    self.name = name
    self.interval = 1 / rate
    self.last_monitor_time = time.monotonic()
    self.next_frame_time = self.last_monitor_time + self.interval
    self._remaining = 0
    self.print_delay_threshold = max(0.0, print_delay_threshold)
    self.frame = 0

  def keep_time(self) -> bool:
    lagged = self.monitor_time()
    if self._remaining > 0: time.sleep(self._remaining)
    return lagged

  def monitor_time(self) -> bool:
    self.frame += 1
    self.last_monitor_time = time.monotonic()
    self._remaining = self.next_frame_time - self.last_monitor_time
    lagged = self._remaining < 0
    if lagged:
      if (self.print_delay_threshold > 0 and self._remaining < -self.print_delay_threshold):
        print(f"{self.name} lagging by {-self._remaining * 1000:.2f}ms")
      self.next_frame_time = self.last_monitor_time + self.interval
    else:
      self.next_frame_time += self.interval
    return lagged


