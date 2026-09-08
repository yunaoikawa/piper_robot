"""Small mobile UI for selecting one lid in an immutable head frame."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import cv2
import numpy as np

from rollout.tapped_lid_target import TappedTarget, frame_sha256


@dataclass
class TapSelectionStore:
    image_bgr: np.ndarray
    timestamp: float

    def __post_init__(self):
        self.frame_hash = frame_sha256(self.image_bgr)
        self.selection: TappedTarget | None = None
        self.event = threading.Event()

    def select(self, payload: dict) -> TappedTarget:
        if payload.get("frame_sha256") != self.frame_hash:
            raise ValueError("表示後に画像が変わりました。再読み込みしてください")
        selection = TappedTarget(
            (float(payload["u"]), float(payload["v"])),
            self.frame_hash,
            self.timestamp,
        )
        self.selection = selection
        self.event.set()
        return selection


def _page(store: TapSelectionStore) -> bytes:
    ok, encoded = cv2.imencode(".jpg", store.image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("failed to encode head image")
    image = base64.b64encode(encoded).decode("ascii")
    return f"""<!doctype html><html lang=ja><meta name=viewport content='width=device-width,initial-scale=1'>
<title>蓋を選択</title><style>
body{{margin:0;background:#111;color:#fff;font-family:-apple-system,sans-serif}}
main{{max-width:900px;margin:auto;padding:12px}} img{{width:100%;height:auto;display:block;touch-action:manipulation}}
#status{{font-size:18px;padding:12px 0}} button{{font-size:20px;width:100%;padding:14px;margin-top:10px}}
</style><main><h2>掴む蓋の青い印を1回タップ</h2><div style='position:relative'>
<img id=frame src='data:image/jpeg;base64,{image}'><canvas id=mark style='position:absolute;inset:0;width:100%;height:100%;pointer-events:none'></canvas>
</div><div id=status>まだ選択されていません</div><button id=go disabled>この蓋を掴む</button></main>
<script>
const hash={json.dumps(store.frame_hash)}, img=document.querySelector('#frame'), canvas=document.querySelector('#mark');
let uv=null; img.onclick=e=>{{const r=img.getBoundingClientRect();uv=[(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height];
canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;let c=canvas.getContext('2d');c.clearRect(0,0,canvas.width,canvas.height);
c.strokeStyle='#ff0';c.lineWidth=5;c.beginPath();c.arc(uv[0]*canvas.width,uv[1]*canvas.height,22,0,7);c.stroke();
document.querySelector('#status').textContent=`選択: ${{uv[0].toFixed(3)}}, ${{uv[1].toFixed(3)}}`;document.querySelector('#go').disabled=false;}};
document.querySelector('#go').onclick=async()=>{{let r=await fetch('/api/select',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{u:uv[0],v:uv[1],frame_sha256:hash}})}});
let j=await r.json();document.querySelector('#status').textContent=r.ok?'選択を受信しました。画面を閉じてOKです':j.error;}};
</script></html>""".encode()


def make_server(store: TapSelectionStore, host="0.0.0.0", port=8094):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _json(self, code, value):
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/":
                self.send_error(404)
                return
            body = _page(store)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/api/select":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                selected = store.select(json.loads(self.rfile.read(length)))
                self._json(200, selected.to_dict())
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                self._json(400, {"error": str(error)})

    return ThreadingHTTPServer((host, port), Handler)
