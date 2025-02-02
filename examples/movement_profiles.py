import math
import numpy as np
import time

from robot.base.base_controller import Vehicle

def circling_profile():
  T_final = 20
  DT = 0.004
  t = np.linspace(0, T_final, int(T_final / DT) + 1)

  R = 1  # turn radius
  w_path = math.pi / 8  # rad/s
  v_path = R * w_path  # m/s
  vx = v_path * np.cos(w_path * t)
  vy = v_path * np.sin(w_path * t)
  w = w_path * np.zeros_like(t) * 2
  u_3dof = np.stack([vx, vy, w], axis=0)  # (3, t)
  return u_3dof


def square_profile():
  T_final = 12
  DT = 0.004
  t = np.linspace(0, T_final, int(T_final / DT) + 1)

  v_path = 0.5  # m/s

  vx = np.zeros_like(t)
  vy = np.zeros_like(t)
  w = np.zeros_like(t)

  for i in range(len(t)):
    if t[i] < 3:
      vx[i] = v_path
      vy[i] = 0
      w[i] = 0
    elif t[i] < 6:
      vx[i] = 0
      vy[i] = v_path
      w[i] = 0
    elif t[i] < 9:
      vx[i] = -v_path
      vy[i] = 0
      w[i] = 0
    else:
      vx[i] = 0
      vy[i] = -v_path
      w[i] = 0

  u_3dof = np.stack([vx, vy, w], axis=0)  # (3, t)
  return u_3dof



if __name__ == "__main__":
  vehicle = Vehicle()
  profiles = square_profile()
  vehicle.start_control()
  try:
    for _ in range(100):
      vehicle.set_target_velocity(np.array([0, 0.1, 0]))
      time.sleep(0.1)
  finally:
    vehicle.stop_control()
