import json
import socket
import time
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np

from rollout.head_stream import HeadCameraStreamServer


class HeadStreamTest(unittest.TestCase):
    def test_health_and_index(self):
        frame = np.zeros((12, 8, 3), dtype=np.uint8)
        server = HeadCameraStreamServer(
            lambda: frame, host="127.0.0.1", port=0,
            status_provider=lambda: "LIVE TEST", fps=30,
        )
        self.assertTrue(server.start())
        try:
            root = f"http://127.0.0.1:{server.port}"
            with urlopen(root + "/healthz", timeout=2) as response:
                self.assertEqual(
                    json.load(response), {"ok": True, "status": "LIVE TEST"}
                )
            with urlopen(root + "/", timeout=2) as response:
                page = response.read().decode()
                self.assertIn('src="/stream.mjpg"', page)
                self.assertIn("LIVE TEST", page)
        finally:
            server.stop()

    def test_token_protects_every_route(self):
        server = HeadCameraStreamServer(
            lambda: None, host="127.0.0.1", port=0, token="secret value"
        )
        self.assertTrue(server.start())
        try:
            root = f"http://127.0.0.1:{server.port}"
            with self.assertRaises(HTTPError) as caught:
                urlopen(root + "/healthz", timeout=2)
            self.assertEqual(caught.exception.code, 401)

            with urlopen(root + "/?token=secret%20value", timeout=2) as response:
                page = response.read().decode()
                self.assertIn("/stream.mjpg?token=secret%20value", page)
        finally:
            server.stop()

    def test_retries_after_port_becomes_available(self):
        blocker = socket.socket()
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        port = blocker.getsockname()[1]
        server = HeadCameraStreamServer(
            lambda: None, host="127.0.0.1", port=port, retry_interval=0.05
        )
        self.assertFalse(server.start())
        blocker.close()
        try:
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                try:
                    with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=0.2) as response:
                        self.assertFalse(json.load(response)["ok"])
                        break
                except OSError:
                    time.sleep(0.05)
            else:
                self.fail("head stream did not acquire the released port")
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
