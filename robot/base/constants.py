import math

# RPC
BASE_RPC_HOST = '10.19.131.241' # tailscale ip
BASE_RPC_PORT = 32000

# policy
POLICY_CONTROL_FREQ = 10
POLICY_CONTROL_PERIOD = 1.0 / POLICY_CONTROL_FREQ

# Vehicle
CONTROL_FREQ = 250  # 250 Hz
CONTROL_PERIOD = 1.0 / CONTROL_FREQ  # 4 ms
NUM_SWERVES = 4
LENGTH = 0.106  # m
WIDTH = 0.152  # m
TIRE_RADIUS = 0.0508  # m

# Encoder magnet offsets
ENCODER_MAGNET_OFFSETS = [-2896.0 / 4096, -3762.0 / 4096, -4794.0 / 4096, 417.0 / 4096]

# Swerve
TWO_PI = 2 * math.pi
N_r1 = 50.0 / 16.0  # Drive gear ratio (1st stage)
N_r2 = 19.0 / 25.0  # Drive gear ratio (2nd stage)
N_w = 45.0 / 15.0  # Wheel gear ratio
DRIVE_GEAR_RATIO = N_r1 * N_r2 * N_w
