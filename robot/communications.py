import time
import zmq
from typing import Any, List, Callable
from robot.msgs import Serializable, s, ns

# Networking Constants
ROBOT_IP = "100.96.33.32"  # tailscale ip
COMMAND_PORT = 5555
ROBOT_STATE_PORT = 5556
VIZ_PORT = 8000
BASE_CAMERA_PORT = 9000

class FrequencyTimer:
    def __init__(self, name: str, frequency: int, delay_warn_threshold: s = s(-1)):
        self.name = name
        self.interval = ns(int(1e9 / frequency))
        self.last_time = ns(time.perf_counter_ns())
        self.delay_warn_threshold = ns(int(delay_warn_threshold * 1e9))

    def __enter__(self):
        self.last_time = ns(time.perf_counter_ns())

    def __exit__(self, *args):
        elapsed = time.perf_counter_ns() - self.last_time
        if elapsed < self.interval:
            time.sleep((self.interval - elapsed)/1e9)
        elif self.delay_warn_threshold > 0 and elapsed > self.interval + self.delay_warn_threshold:
            print(f"{self.name} is running behind by {(self.interval - elapsed)/1e6} ms")


class Publisher:
    def __init__(self, ctx: zmq.Context, port: int, host: str = "*", HWM: int | None = None):
        self.socket: zmq.Socket = ctx.socket(zmq.PUB)
        self.socket.bind(f"tcp://{host}:{port}")
        if HWM is not None:
            self.socket.setsockopt(zmq.SNDHWM, HWM)

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
        deserializer: List[Callable[[bytes], Serializable]],
        host: str = "localhost",
        conflate: bool = True,
        no_block: bool = False,
    ):
        self.socket: zmq.Socket = ctx.socket(zmq.SUB)
        self.socket.connect(f"tcp://{host}:{port}")
        for topic in topics:
            self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        if conflate:
            self.socket.setsockopt(zmq.CONFLATE, 1)

        self.deserializer = {topic: deserializer[i] for i, topic in enumerate(topics)}
        self.no_block = zmq.NOBLOCK if no_block else 0

    def receive(self) -> tuple[str | None, Any | None]:
        try:
            payload = self.socket.recv_multipart(flags=self.no_block)
            topic = payload[0].decode("utf-8")
            data = self.deserializer[topic](payload[1])
            return topic, data
        except zmq.error.Again:
            return None, None

    def register_poller(self, poller: zmq.Poller):
        poller.register(self.socket, zmq.POLLIN)

    def stop(self):
        self.socket.close()
