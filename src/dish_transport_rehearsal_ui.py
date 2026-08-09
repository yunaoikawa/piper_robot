#!/usr/bin/env python3
"""Phone UI for photographic approval of air-transport checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import cv2
import numpy as np


@dataclass
class CheckpointApprovalStore:
    """Thread-safe current checkpoint and one-shot operator decision."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    event: threading.Event = field(default_factory=threading.Event)
    revision: int = 0
    state: dict = field(default_factory=lambda: {"status": "waiting_for_robot"})
    images: dict[str, bytes] = field(default_factory=dict)
    decision: str | None = None

    def publish(
        self,
        *,
        segment: str,
        checkpoint: str,
        physical_arm: str,
        metrics: dict,
        head_bgr: np.ndarray,
        wrist_bgr: np.ndarray,
        continue_allowed: bool = True,
    ) -> int:
        encoded = {}
        for name, image in (("head", head_bgr), ("wrist", wrist_bgr)):
            ok, data = cv2.imencode(
                ".jpg", np.asarray(image), [cv2.IMWRITE_JPEG_QUALITY, 93]
            )
            if not ok:
                raise RuntimeError(f"failed to encode {name} checkpoint image")
            encoded[name] = data.tobytes()
        with self.lock:
            self.revision += 1
            self.decision = None
            self.event.clear()
            self.images = encoded
            self.state = {
                "status": "awaiting_operator",
                "revision": self.revision,
                "segment": str(segment),
                "checkpoint": str(checkpoint),
                "physical_arm": str(physical_arm),
                "metrics": metrics,
                "continue_allowed": bool(continue_allowed),
                "published_at_s": time.time(),
            }
            return self.revision

    def decide(self, revision: int, decision: str) -> dict:
        if decision not in {"continue", "abort_hold", "abort_home"}:
            raise ValueError("unsupported decision")
        with self.lock:
            if int(revision) != self.revision:
                raise ValueError("古い確認画面です。再読み込みしてください")
            if self.state.get("status") != "awaiting_operator":
                raise ValueError("この停止点はすでに処理済みです")
            if decision == "continue" and not self.state.get("continue_allowed", False):
                raise ValueError("水平姿勢ゲートが不合格です。保持またはホームを選んでください")
            self.decision = decision
            self.state = {**self.state, "status": "decision_received", "decision": decision}
            self.event.set()
            return dict(self.state)

    def wait(self, timeout_s: float | None = None) -> str:
        if not self.event.wait(timeout_s):
            raise TimeoutError("operator checkpoint approval timed out")
        with self.lock:
            if self.decision is None:
                raise RuntimeError("checkpoint event fired without a decision")
            return self.decision

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.state)

    def image(self, name: str) -> bytes | None:
        with self.lock:
            return self.images.get(name)


HTML = """<!doctype html><html lang=ja><meta charset=utf-8>
<meta name=viewport content='width=device-width,initial-scale=1'>
<title>水平搬送チェック</title><style>
body{margin:0;background:#101114;color:#f4f4f4;font-family:-apple-system,sans-serif}
main{max-width:920px;margin:auto;padding:12px}.card{background:#202228;border-radius:14px;padding:12px;margin:10px 0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}img{width:100%;height:auto;background:#000;border-radius:9px}
button{width:100%;font-size:19px;padding:14px;margin:6px 0;border:0;border-radius:11px;color:white;background:#2677ff}
.hold{background:#d98b18}.home{background:#c83232}pre{white-space:pre-wrap;word-break:break-word;font-size:12px}
@media(max-width:650px){.grid{grid-template-columns:1fr}}
</style><main><h2>皿の水平エアー搬送</h2><div class=card id=status>接続中…</div>
<div class=grid><div><b>Head</b><img id=head></div><div><b id=wristName>Wrist</b><img id=wrist></div></div>
<div class=card><pre id=metrics></pre></div>
<button id=go>写真と姿勢を確認：次へ</button>
<button class=hold id=hold>中止して現在位置で保持</button>
<button class=home id=home>中止してホームへ戻す</button></main><script>
let current=null;
async function refresh(){let r=await fetch('/api/state',{cache:'no-store'}),s=await r.json();
current=s;document.querySelector('#status').textContent=s.status==='awaiting_operator'?`${s.segment} / ${s.checkpoint} / ${s.physical_arm}手`:`状態: ${s.status}`;
document.querySelector('#metrics').textContent=JSON.stringify(s.metrics||{},null,2);
document.querySelector('#wristName').textContent=(s.physical_arm||'')+' wrist';
if(s.revision){document.querySelector('#head').src='/image/head?r='+s.revision;document.querySelector('#wrist').src='/image/wrist?r='+s.revision}
let enabled=s.status==='awaiting_operator';document.querySelector('#go').disabled=!enabled||!s.continue_allowed;
for(let id of ['hold','home'])document.querySelector('#'+id).disabled=!enabled}
async function decide(decision){if(!current||!current.revision)return;let r=await fetch('/api/decision',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({revision:current.revision,decision})});let j=await r.json();if(!r.ok)alert(j.error);await refresh()}
document.querySelector('#go').onclick=()=>decide('continue');document.querySelector('#hold').onclick=()=>decide('abort_hold');document.querySelector('#home').onclick=()=>decide('abort_home');
refresh();setInterval(refresh,1500);</script></html>"""


def handler_for(store: CheckpointApprovalStore):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _send(self, status: int, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("cache-control", "no-store")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value, status=HTTPStatus.OK):
            self._send(
                int(status),
                json.dumps(value, ensure_ascii=False).encode(),
                "application/json; charset=utf-8",
            )

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/?"):
                return self._send(200, HTML.encode(), "text/html; charset=utf-8")
            if self.path == "/api/state":
                return self._json(store.snapshot())
            for name in ("head", "wrist"):
                if self.path.startswith(f"/image/{name}"):
                    image = store.image(name)
                    if image is None:
                        return self._json({"error": "image not ready"}, 404)
                    return self._send(200, image, "image/jpeg")
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/api/decision":
                return self._json({"error": "not found"}, 404)
            try:
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                self._json(store.decide(payload["revision"], payload["decision"]))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                self._json({"error": str(error)}, 409)

    return Handler


def make_server(
    store: CheckpointApprovalStore,
    *,
    host: str = "0.0.0.0",
    port: int = 8097,
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, int(port)), handler_for(store))
