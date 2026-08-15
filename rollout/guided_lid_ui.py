"""Low-latency phone UI for deterministic guided lid collection."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from urllib.parse import parse_qs, urlparse

import cv2


HTML = r"""<!doctype html><html lang=ja><head><meta name=viewport content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no"><style>
body{font:16px system-ui;margin:0;background:#101114;color:#eee}main{max-width:760px;margin:auto;padding:10px}.row{display:flex;gap:7px;margin:8px 0;flex-wrap:wrap}button,input,select{font:inherit;font-weight:650;min-height:48px;border:0;border-radius:9px;padding:8px 12px}button{background:#2878ff;color:#fff}.danger{background:#bd1732}.ok{background:#17864b}.auto{background:#8250df}.axis{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.axis button{font-size:17px}.cams{display:grid;grid-template-columns:1fr 1fr;gap:5px}.cams img{width:100%;background:#222;min-height:80px}.cams img:first-child{grid-column:1/3}pre{white-space:pre-wrap;background:#222;padding:8px;border-radius:8px;max-height:40vh;overflow:auto}small{color:#bbb}.value{width:76px;background:#eee;color:#111}</style></head><body><main>
<h3>ACTなし・ユーザー誘導 蓋デモ収集</h3>
<div class=row><button onclick="cmd('home')">HOME</button><button class=ok onclick="cmd('start')">試行開始</button><button class=auto onclick="cmd('enable_auto')">AUTO開始</button><button class=danger onclick="cmd('stop')">STOP</button></div>
<div class=cams><img id=head alt="head"><img id=left alt="left"><img id=right alt="right"></div>
<div class=row><label>補正量 <select id=step><option value=1>1 mm</option><option value=5 selected>5 mm</option><option value=10>10 mm</option><option value=20>20 mm</option></select></label><label>任意mm <input class=value id=custom type=number step=1 placeholder=mm></label></div>
<small>補正は次の試行だけに適用。低位置の腕を直接ジョグしません。物理右=robot Y−、下=Z−。</small>
<div class=axis><button onclick="n('x',-1)">奥 X−</button><button onclick="n('y',-1)">右 Y−</button><button onclick="n('z',-1)">下 Z−</button><button onclick="n('x',1)">手前 X＋</button><button onclick="n('y',1)">左 Y＋</button><button onclick="n('z',1)">上 Z＋</button></div>
<div class=row><button class=ok onclick="cmd('success')">SUCCESS</button><select id=reason><option>grasp_miss</option><option>jam</option><option>drop</option><option>wrong_placement</option><option>abort</option></select><button class=danger onclick="cmd('failure',{reason:reason.value})">FAIL</button></div>
<pre id=status>connecting…</pre>
<script>const token=new URLSearchParams(location.search).get('token')||'';async function cmd(command,payload={}){try{let r=await fetch('/api/command?token='+encodeURIComponent(token),{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({command,payload})});status.textContent=JSON.stringify(await r.json(),null,2)}catch(e){status.textContent='通信失敗: '+e}}function n(axis,direction){let raw=custom.value;let mm=raw===''?+step.value:Math.abs(+raw);cmd('adjust',{axis,direction,step_mm:mm})}let es=new EventSource('/events?token='+encodeURIComponent(token));es.onmessage=e=>{status.textContent=JSON.stringify(JSON.parse(e.data),null,2)};const cams=['head','left','right'];let ci=0;function refresh(){let c=cams[ci++%cams.length],im=document.getElementById(c),next=()=>setTimeout(refresh,250);let probe=new Image();probe.onload=()=>{im.src=probe.src;next()};probe.onerror=next;probe.src='/frame/'+c+'.jpg?token='+encodeURIComponent(token)+'&t='+Date.now()}refresh();</script></main></body></html>"""


class GuidedLidUI:
    def __init__(self, host, port, token, snapshot_provider, command_callback, frame_provider):
        self.host = host
        self.port = int(port)
        self.token = token or ""
        self.snapshot_provider = snapshot_provider
        self.command_callback = command_callback
        self.frame_provider = frame_provider
        self.server = None
        self.thread = None
        self.stopped = threading.Event()

    def start(self) -> None:
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
                    last = None
                    try:
                        while not owner.stopped.is_set():
                            snapshot = owner.snapshot_provider()
                            encoded = json.dumps(snapshot, separators=(",", ":"))
                            if encoded != last:
                                self.wfile.write(f"data:{encoded}\n\n".encode())
                                self.wfile.flush()
                                last = encoded
                            time.sleep(0.05)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                if path.startswith("/frame/") and path.endswith(".jpg"):
                    camera = path.rsplit("/", 1)[-1][:-4]
                    frame, _ = owner.frame_provider(camera)
                    if frame is None:
                        return self.send_bytes(b"camera unavailable", "text/plain", 503)
                    # Phone UI is for rapid operator review, not recording.  Keep
                    # acquisition full-resolution while sending a small snapshot.
                    height, width = frame.shape[:2]
                    if width > 480:
                        scale = 480.0 / width
                        frame = cv2.resize(
                            frame, (480, max(1, int(height * scale))),
                            interpolation=cv2.INTER_AREA,
                        )
                    ok, encoded = cv2.imencode(
                        ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 58],
                    )
                    if not ok:
                        return self.send_bytes(b"encode failed", "text/plain", 500)
                    return self.send_bytes(encoded.tobytes(), "image/jpeg")
                if path.startswith("/stream/") and path.endswith(".mjpg"):
                    camera = path.rsplit("/", 1)[-1][:-5]
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    previous = None
                    try:
                        while not owner.stopped.is_set():
                            frame, frame_id = owner.frame_provider(camera)
                            if frame is None or frame_id == previous:
                                time.sleep(0.015)
                                continue
                            previous = frame_id
                            ok, encoded = cv2.imencode(
                                ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 72],
                            )
                            if not ok:
                                continue
                            payload = encoded.tobytes()
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                + str(len(payload)).encode() + b"\r\n\r\n" + payload + b"\r\n"
                            )
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
                    # The callback only enqueues; HTTP response is not coupled
                    # to robot motion and remains fast on the phone.
                    result = owner.command_callback(body["command"], body.get("payload", {}))
                    self.send_bytes(json.dumps(result).encode(), "application/json")
                except Exception as error:
                    self.send_bytes(
                        json.dumps({"error": str(error)}).encode(),
                        "application/json", HTTPStatus.BAD_REQUEST,
                    )

        self.server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stopped.set()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
