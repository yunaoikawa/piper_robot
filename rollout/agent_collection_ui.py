"""Low-latency phone UI for ACT intervention collection."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from urllib.parse import parse_qs, urlparse

import cv2


HTML = r"""<!doctype html><html><head><meta name=viewport content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no"><style>
body{font:16px system-ui;margin:0;background:#111;color:#eee}main{max-width:720px;margin:auto;padding:10px}.row{display:flex;gap:7px;margin:8px 0;flex-wrap:wrap}button,select{font:inherit;font-weight:650;min-height:48px;border:0;border-radius:9px;padding:8px 14px}button{background:#2878ff;color:white}.warn{background:#c03}.ok{background:#198754}.axis{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.axis button{font-size:19px}.cams{display:grid;grid-template-columns:1fr 1fr;gap:5px}.cams img{width:100%;background:#222}.cams img:first-child{grid-column:1/3}pre{white-space:pre-wrap;background:#222;padding:8px;border-radius:8px}small{color:#bbb}</style></head><body><main>
<div class=row><button onclick="cmd('home')">HOME</button><button onclick="cmd('start')">START</button><button class=warn onclick="cmd('pause')">PAUSE</button><button class=ok onclick="cmd('resume')">RESUME</button></div>
<small>画像をタップして対象を選択（開始前）</small><div class=cams><img id=head src="/stream/head.mjpg"><img src="/stream/left.mjpg"><img src="/stream/right.mjpg"></div>
<div class=row><select id=step><option value=1>1 mm</option><option value=2 selected>2 mm</option><option value=5>5 mm</option></select></div>
<small>物理方向（PAUSE中は把持姿勢を直接ジョグ）</small><div class=axis><button onclick="n('x',-1)">後ろ X−</button><button onclick="n('y',-1)">右 Y−</button><button onclick="n('z',-1)">下 Z−</button><button onclick="n('x',1)">前 X＋</button><button onclick="n('y',1)">左 Y＋</button><button onclick="n('z',1)">上 Z＋</button></div>
<div class=row><button class=ok onclick="cmd('success')">SUCCESS</button><select id=reason><option>grasp_miss</option><option>jam</option><option>drop</option><option>wrong_placement</option><option>abort</option></select><button class=warn onclick="cmd('failure',{reason:reason.value})">FAIL</button></div><pre id=status>connecting…</pre>
<script>const token=new URLSearchParams(location.search).get('token')||'';async function cmd(command,payload={}){let r=await fetch('/api/command?token='+encodeURIComponent(token),{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({command,payload})});status.textContent=JSON.stringify(await r.json(),null,2)}function n(axis,direction){cmd('nudge',{axis,direction,step_mm:+step.value})}head.onclick=e=>{let r=head.getBoundingClientRect();cmd('select_target',{u:(e.clientX-r.left)/r.width,v:(e.clientY-r.top)/r.height,frame_id:head.dataset.frame||null})};let es=new EventSource('/events?token='+encodeURIComponent(token));es.onmessage=e=>{let s=JSON.parse(e.data);status.textContent=JSON.stringify(s,null,2)};</script></main></body></html>"""


class AgentCollectionUI:
    def __init__(self, host, port, token, state, command_callback, frame_provider):
        self.host, self.port, self.token = host, int(port), token or ""
        self.state = state
        self.command_callback = command_callback
        self.frame_provider = frame_provider
        self.server = None
        self.thread = None

    def start(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                return

            def authorized(self):
                supplied = parse_qs(urlparse(self.path).query).get("token", [""])[0]
                return not owner.token or supplied == owner.token

            def send_bytes(self, payload, content_type, status=200):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/":
                    return self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
                if not self.authorized():
                    return self.send_bytes(b"unauthorized", "text/plain", 401)
                if path == "/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    revision = -1
                    try:
                        while not owner.state.latest_metrics.get("stopped"):
                            snapshot = owner.state.snapshot()
                            if snapshot["revision"] != revision:
                                blob = json.dumps(snapshot, separators=(",", ":"))
                                self.wfile.write(f"data:{blob}\n\n".encode())
                                self.wfile.flush()
                                revision = snapshot["revision"]
                            time.sleep(0.05)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                if path.startswith("/stream/") and path.endswith(".mjpg"):
                    camera = path.rsplit("/", 1)[-1][:-5]
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    previous = None
                    try:
                        while True:
                            frame, frame_id = owner.frame_provider(camera)
                            if frame is None or frame_id == previous:
                                time.sleep(0.015); continue
                            previous = frame_id
                            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 72])
                            if not ok: continue
                            payload = encoded.tobytes()
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload + b"\r\n")
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                self.send_error(404)

            def do_POST(self):
                if urlparse(self.path).path != "/api/command":
                    return self.send_error(404)
                if not self.authorized():
                    return self.send_bytes(b'{"error":"unauthorized"}', "application/json", 401)
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(length) or b"{}")
                    result = owner.command_callback(body["command"], body.get("payload", {}))
                    self.send_bytes(json.dumps(result).encode(), "application/json")
                except Exception as error:
                    self.send_bytes(json.dumps({"error": str(error)}).encode(), "application/json", HTTPStatus.BAD_REQUEST)

        self.server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        if self.server:
            self.state.latest_metrics["stopped"] = True
            self.server.shutdown()
            self.server.server_close()
        if self.thread:
            self.thread.join(timeout=2)
