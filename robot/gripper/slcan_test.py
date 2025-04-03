import can

# Replace with your actual device port
slcan_interface = can.Bus(
    interface='slcan',
    channel='/dev/ttyACM4@921600',  # or 'COM3' on Windows
    bitrate=1000000
)

msg = can.Message(arbitration_id=0x001, data=[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfc], is_extended_id=False)

slcan_interface.send(msg)
print("Message sent successfully")


msg = slcan_interface.recv(timeout=5.0)
if msg:
    print(f"Received message: {msg}")
else:
    print("No message received")


slcan_interface.shutdown()
