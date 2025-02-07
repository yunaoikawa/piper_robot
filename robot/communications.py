import zmq
from typing import Any, List, Callable, Tuple

# Networking Constants
ROBOT_IP = "100.96.33.32"  # tailscale ip
COMMAND_PORT = 5555
ROBOT_STATE_PORT = 5556


class Publisher:
  def __init__(self, ctx: zmq.Context, port: int, serializer: Callable[[Any], Tuple[str, bytes]], host: str = "*"):
    self.socket: zmq.Socket = ctx.socket(zmq.PUB)
    self.socket.bind(f"tcp://{host}:{port}")
    self.serializer = serializer

  def publish(self, topic: str, data: Any, copy: bool = True):
    metadata, data = self.serializer(data)
    self.socket.send_string(topic, zmq.SNDMORE)
    self.socket.send_string(metadata, zmq.SNDMORE)
    self.socket.send(data, copy=copy)

  def stop(self):
    self.socket.close()


class Subscriber:
  def __init__(self, ctx: zmq.Context, port: int, topics: List[str], deserializer: callable, host: str = "localhost", conflate: bool = True):
    self.socket: zmq.Socket = ctx.socket(zmq.SUB)
    self.socket.connect(f"tcp://{host}:{port}")
    for topic in topics:
      self.socket.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
    if conflate:
      self.socket.setsockopt(zmq.CONFLATE, 1)
    self.deserializer = deserializer

  def receive(self) -> tuple[str, Any]:
    topic, metadata, data = self.socket.recv_multipart()
    return topic.decode("utf-8"), self.deserializer(metadata.decode("utf-8"), data)

  def register_poller(self, poller: zmq.Poller):
    poller.register(self.socket, zmq.POLLIN)

  def stop(self):
    self.socket.close()
