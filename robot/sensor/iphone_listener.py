import time
import zmq
import cv2
from robot.communications import Subscriber, BASE_CAMERA_PORT
from robot.msgs import EncodedImage, EncodedDepth, Pose


def main():
    ctx = zmq.Context()
    subscriber = Subscriber(
        ctx,
        BASE_CAMERA_PORT,
        ["/base/image", "/base/depth", "/base/pose"],
        [EncodedImage.deserialize, EncodedDepth.deserialize, Pose.deserialize],
    )

    while True:
        topic, data = subscriber.receive()
        if topic == "/base/image":
            current_time = time.perf_counter_ns()
            img = cv2.imdecode(data.image, cv2.IMREAD_COLOR)
            print(img.shape)
            delay_ms = (current_time - data.timestamp) / 1_000_000
            print(f"Received image with delay {delay_ms:.2f}ms")
        elif topic == "/base/depth":
            current_time = time.perf_counter_ns()
            delay_ms = (current_time - data.timestamp) / 1_000_000
        elif topic == "/base/pose":
            current_time = time.perf_counter_ns()
            delay_ms = (current_time - data.timestamp) / 1_000_000


if __name__ == "__main__":
    main()