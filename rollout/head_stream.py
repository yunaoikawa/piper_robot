"""Small, dependency-free HTTP server for viewing a live camera frame.

The server intentionally exposes video only: there are no robot-control HTTP
endpoints.  Bind it to the robot's Tailscale address (or put Tailscale Serve in
front of a loopback bind) to keep the feed off the public Internet.
"""

from __future__ import annotations

import html
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import parse_qs, quote, urlsplit

import cv2
import numpy as np


FrameProvider = Callable[[], np.ndarray | None]
StatusProvider = Callable[[], str]


class _StreamHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, *, frame_provider, status_provider,
                 token, fps, jpeg_quality):
        super().__init__(address, handler)
        self.frame_provider = frame_provider
        self.status_provider = status_provider
        self.token = token
        self.frame_period = 1.0 / max(1.0, float(fps))
        self.jpeg_quality = int(np.clip(jpeg_quality, 20, 100))


class _StreamHandler(BaseHTTPRequestHandler):
    server: _StreamHTTPServer

    def log_message(self, fmt, *args):
        # Avoid one access-log line per reconnect on the robot console.
        return

    def _authorized(self, query):
        expected = self.server.token
        return not expected or parse_qs(query).get("token", [None])[0] == expected

    def _deny(self):
        body = b"Unauthorized\n"
        self.send_response(401)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlsplit(self.path)
        if not self._authorized(parsed.query):
            self._deny()
            return

        if parsed.path == "/stream.mjpg":
            self._stream()
        elif parsed.path == "/healthz":
            self._health()
        elif parsed.path in ("/", "/index.html"):
            self._index(parsed.query)
        else:
            self.send_error(404)

    def _index(self, query):
        token = parse_qs(query).get("token", [None])[0]
        stream_url = "/stream.mjpg"
        if token:
            stream_url += "?token=" + quote(token, safe="")
        status = html.escape(self.server.status_provider())
        page = f"""<!doctype html>
<html><head><meta name="viewport" content="width=device-width,initial-scale=1,
maximum-scale=1,user-scalable=no"><title>Robot head camera</title>
<style>html,body{{margin:0;width:100%;height:100%;background:#000;color:#fff;
font-family:-apple-system,sans-serif;overflow:hidden}}img{{width:100%;height:100%;
object-fit:contain}}.status{{position:fixed;left:10px;top:10px;padding:6px 9px;
background:#0009;border-radius:6px;font-size:13px}}</style></head>
<body><img src="{stream_url}" alt="Robot head camera"><div class="status">{status}</div>
</body></html>""".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(page)

    def _health(self):
        frame = self.server.frame_provider()
        body = json.dumps({
            "ok": frame is not None and getattr(frame, "size", 0) > 0,
            "status": self.server.status_provider(),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        try:
            while True:
                started = time.monotonic()
                rgb = self.server.frame_provider()
                if rgb is None or getattr(rgb, "size", 0) == 0:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Waiting for head camera...", (105, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    # Record3D frames are RGB and mounted in portrait orientation.
                    frame = cv2.rotate(np.asarray(rgb), cv2.ROTATE_90_CLOCKWISE)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                status = self.server.status_provider()
                if status:
                    cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                ok, encoded = cv2.imencode(
                    ".jpg", frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.server.jpeg_quality],
                )
                if ok:
                    payload = encoded.tobytes()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode())
                    self.wfile.write(payload)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()

                delay = self.server.frame_period - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return


class HeadCameraStreamServer:
    """Serve a Record3D RGB frame provider as a Safari-compatible MJPEG page."""

    def __init__(self, frame_provider: FrameProvider, *, host="0.0.0.0", port=8080,
                 status_provider: StatusProvider | None = None, token: str | None = None,
                 fps=15.0, jpeg_quality=75, retry_interval=2.0):
        self.frame_provider = frame_provider
        self.host = host
        self.port = port
        self.status_provider = status_provider or (lambda: "LIVE")
        self.token = token
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.retry_interval = retry_interval
        self._server = None
        self._thread = None
        self._retry_thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def start(self):
        self._stop_event.clear()
        if self._start_server(log_error=True):
            return True
        if self.retry_interval and not self._retry_thread:
            print(f"[head-stream] retrying every {self.retry_interval:g}s")
            self._retry_thread = threading.Thread(
                target=self._retry_loop, name="head-camera-http-retry", daemon=True
            )
            self._retry_thread.start()
        return False

    def _start_server(self, *, log_error):
        try:
            server = _StreamHTTPServer(
                (self.host, self.port), _StreamHandler,
                frame_provider=self.frame_provider,
                status_provider=self.status_provider,
                token=self.token,
                fps=self.fps,
                jpeg_quality=self.jpeg_quality,
            )
        except OSError as exc:
            if log_error:
                print(f"[head-stream] could not listen on {self.host}:{self.port}: {exc}")
            return False

        with self._lock:
            if self._stop_event.is_set():
                server.server_close()
                return False
            if self._server is not None:
                server.server_close()
                return True
            self._server = server
            self.port = server.server_address[1]
            self._thread = threading.Thread(
                target=server.serve_forever, name="head-camera-http", daemon=True
            )
            self._thread.start()
        suffix = "/?token=<token>" if self.token else "/"
        print(f"[head-stream] open http://{self.host}:{self.port}{suffix} on the tailnet")
        return True

    def _retry_loop(self):
        while not self._stop_event.wait(self.retry_interval):
            if self._start_server(log_error=False):
                return

    def stop(self):
        self._stop_event.set()
        with self._lock:
            server = self._server
            thread = self._thread
            retry_thread = self._retry_thread
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2.0)
        if retry_thread is not None:
            retry_thread.join(timeout=2.0)
        with self._lock:
            self._server = None
            self._thread = None
            self._retry_thread = None
