#!/bin/bash

sudo ip link set up can_left type can bitrate 1000000
sleep 0.1

sudo ip link set up can_right type can bitrate 1000000
sleep 0.1

python3 home_gripper.py
sleep 0.1

python3 piper_reset.py
