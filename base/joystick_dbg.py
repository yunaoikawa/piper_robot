import pygame
import time

pygame.init()

# Set up the joystick
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)

# Print the joystick's name
print(joystick.get_name())

while True:
    # Get the joystick's state
    pygame.event.pump()
    print(joystick.get_axis(0), joystick.get_axis(1))
    print(joystick.get_button(0), joystick.get_button(1))
    print()
    time.sleep(0.1)
