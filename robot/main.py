from multiprocessing import Process
import signal

from robot.controller.controller_node import main as launch_controller_node
from robot.sensor.iphone import main as launch_iphone_node

nodes = [
    (launch_controller_node, None),
    (launch_iphone_node, (0,))
]
processes: list[Process] = []

def sigint_handler(signum, frame):
  for p in processes: p.terminate()
  print("Robot server shutdown gracefully.")
  exit(0)

signal.signal(signal.SIGINT, sigint_handler)

if __name__ == "__main__":

  for node, args in nodes:
    p = Process(target=node) if args is None else Process(target=node, args=args)
    p.start()
    processes.append(p)

  for p in processes: p.join()