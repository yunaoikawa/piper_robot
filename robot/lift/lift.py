import os

os.environ["CTR_TARGET"] = "Hardware"
import phoenix6.unmanaged
from phoenix6 import configs, controls, hardware, signals


class Lift:
    def __init__(self):
        self.lift_motor = hardware.TalonFX(1, canbus="Lift")

def main():
    lift = Lift()

    print("Lift initialized")

    print(lift.lift_motor)


if __name__ == "__main__":
    main()
