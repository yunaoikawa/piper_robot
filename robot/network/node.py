import lcm
import threading
import signal
from typing import Any, Callable
import sys
import time
class Node:
    def __init__(self):
        self.lc = lcm.LCM()
        self.stop_event = threading.Event()
        self.threads = []
        self.subscriptions = []

    def create_thread(self, callback: Callable):
        t = threading.Thread(target=callback)
        self.threads.append(t)
        t.start()

    def publish(self, channel: str, msg: Any):
        self.lc.publish(channel, msg.encode())

    def subscribe(self, channel: str, handler: Callable):
        s = self.lc.subscribe(channel, handler)
        self.subscriptions.append(s)
        return s

    def stop(self):
        pass

    def check_timestamp(self, timestamp: int, max_delay: float = 0.1) -> bool:
        current_time = time.perf_counter_ns()
        delay = (current_time - timestamp) / 1e9
        if delay > max_delay or delay < 0:
            print(f"Skipping message because of delay: {delay}s")
            return False
        return True

    def spin(self):
        def signal_handler(sig, frame):
            print('Ctrl+C detected! Stopping...')
            self.stop_event.set()
            for t in self.threads:
                t.join()
            for s in self.subscriptions:
                self.lc.unsubscribe(s)
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        while not self.stop_event.is_set():
            self.lc.handle_timeout(1000)


