import time
import numpy as np
from constants import POLICY_CONTROL_PERIOD


from base_server import BaseManager
from constants import BASE_RPC_HOST, BASE_RPC_PORT
from robot_secrets import RPC_AUTH_KEY

manager = BaseManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTH_KEY)
manager.connect()
base = manager.Base()
try:
    base.reset()
    for i in range(50):
        base.execute_action({'v': np.array([0.0, 0.1, 0.0])})
        # print(base.get_state())
        time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
finally:
    base.close()