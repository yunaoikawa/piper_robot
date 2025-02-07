from multiprocessing import Process
import signal

from robot.controller.controller_node import launch_controller_node

nodes = [launch_controller_node]
processes: list[Process] = []

def sigint_handler(signum, frame):
  for p in processes: p.terminate()
  print("Robot server shutdown gracefully.")
  exit(0)

signal.signal(signal.SIGINT, sigint_handler)

if __name__ == "__main__":

  for node in nodes:
    p = Process(target=node)
    p.start()
    processes.append(p)

  for p in processes: p.join()
