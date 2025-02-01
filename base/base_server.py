from base_controller import Vehicle
from multiprocessing.managers import BaseManager as MPBaseManager

import time
from constants import BASE_RPC_HOST, BASE_RPC_PORT
from robot_secrets import RPC_AUTH_KEY

class Base:
    def __init__(self, max_vel=np.array((0.5, 0.5, 1.57)), max_accel=np.array((0.25, 0.25, 0.79))):
        self.max_vel = max_vel
        self.max_accel = max_accel
        self.vehicle = None

    def reset(self):
        # Stop low-level control
        if self.vehicle is not None:
            if self.vehicle.control_loop_running:
                self.vehicle.stop_control()

        # Create new instance of vehicle
        self.vehicle = Vehicle(max_vel=self.max_vel, max_accel=self.max_accel)

        # Start low-level control
        self.vehicle.start_control()
        while not self.vehicle.control_loop_running:
            time.sleep(0.01)

    def execute_action(self, action):
        self.vehicle.set_target_velocity(action['v'])

    def get_state(self):
        state = {'base_pose': self.vehicle.x}
        return state

    def close(self):
        if self.vehicle is not None:
            if self.vehicle.control_loop_running:
                self.vehicle.stop_control()

class BaseManager(MPBaseManager):
    pass

BaseManager.register('Base', Base)

if __name__ == "__main__":
    manager = BaseManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTH_KEY)
    server = manager.get_server()
    print(f'Base manager server started at {BASE_RPC_HOST}:{BASE_RPC_PORT}')
    server.serve_forever()
