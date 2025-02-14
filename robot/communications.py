import time
import zmq
from typing import Any, List, Callable
from robot.msgs import Serializable

# Networking Constants
ROBOT_IP = "100.96.33.32"  # tailscale ip
COMMAND_PORT = 5555
ROBOT_STATE_PORT = 5556
BASE_CAMERA_PORT = 9000

class FrequencyTimer:
    def __init__(self, name: str, frequency: int, delay_warn_threshold: int = -1):
        self.name = name
        self.interval = int(1e9 / frequency)
        self.last_time = time.perf_counter_ns()
        self.delay_warn_threshold = delay_warn_threshold

    def __enter__(self):
        self.last_time = time.perf_counter_ns()

    def __exit__(self, *args):
        elapsed = time.perf_counter_ns() - self.last_time
        if elapsed < self.interval:
            time.sleep((self.interval - elapsed)/1e9)
        elif self.delay_warn_threshold > 0 and elapsed > self.interval + self.delay_warn_threshold:
            print(f"{self.name} is running behind by {(self.interval - elapsed)/1e6} ms")


class Publisher:
    def __init__(self, ctx: zmq.Context, port: int, host: str = "*"):
        self.socket: zmq.Socket = ctx.socket(zmq.PUB)
        self.socket.bind(f"tcp://{host}:{port}")

    def publish(self, topic: str, data: Serializable, copy: bool = True):
        payload = data.serialize()
        self.socket.send_string(topic, zmq.SNDMORE)
        self.socket.send(payload, copy=copy)

    def stop(self):
        self.socket.close()


class Subscriber:
    def __init__(
        self,
        ctx: zmq.Context,
        port: int,
        topics: List[str],
        deserializer: Callable[[bytes], Serializable],
        host: str = "localhost",
        conflate: bool = True,
    ):
        self.socket: zmq.Socket = ctx.socket(zmq.SUB)
        self.socket.connect(f"tcp://{host}:{port}")
        for topic in topics:
            self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        if conflate:
            self.socket.setsockopt(zmq.CONFLATE, 1)
        self.deserializer = deserializer

    def receive(self) -> tuple[str, Any]:
        topic, data = self.socket.recv_multipart()
        return topic.decode("utf-8"), self.deserializer(data)

    def register_poller(self, poller: zmq.Poller):
        poller.register(self.socket, zmq.POLLIN)

    def stop(self):
        self.socket.close()
