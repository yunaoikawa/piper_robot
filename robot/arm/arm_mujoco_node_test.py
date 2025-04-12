import lcm
from robot.msgs.pose import Pose

if __name__ == "__main__":
    lc = lcm.LCM()
    pose= Pose()
    pose.translation = [0.19, 0.0, 0.2]
    pose.rotation = [1.0, 0.0, 0.0, 0.0]
    lc.publish("ARM_COMMAND", pose.encode())
    print("published")