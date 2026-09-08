"""Small phone UI for agentic collection status and operator intervention."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from urllib.parse import parse_qs, urlparse


PAGE = """<!doctype html><html lang=ja><meta charset=utf-8>
<meta name=viewport content='width=device-width,initial-scale=1'>
<title>Agentic Collection</title>
<style>body{font-family:-apple-system,sans-serif;background:#111;color:#eee;margin:20px}
button{font-size:20px;padding:14px;margin:5px;border-radius:10px}pre{white-space:pre-wrap;background:#222;padding:12px;border-radius:10px}</style>
<h2>Agentic Data Collection</h2><div id=s></div>
<button onclick="post('/api/hold')">HOLD</button>
<button onclick="post('/api/intervene')">TELEOP TAKEOVER</button>
<button onclick="overrideGate()">不確実判定を承認</button><pre id=j></pre>
<script>async function refresh(){let r=await fetch('/api/status');let x=await r.json();
s.textContent=(x.armed?'ARMED':'NOT ARMED')+' / '+x.mode+' / '+x.task.name+' / '+(x.checkpoint||'idle')+(x.held_uncertain?' / HOLDING FOR REVIEW':'');
j.textContent=JSON.stringify(x,null,2)}async function post(p,b=''){await fetch(p,{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:b});refresh()}
function overrideGate(){let n=prompt('理由');if(n!==null)post('/api/override','note='+encodeURIComponent(n))}
setInterval(refresh,1000);refresh()</script></html>"""


def make_agentic_ui(supervisor, host: str, port: int):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status, value):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/api/status":
                self._json(200, supervisor.status())
                return
            data = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                params = parse_qs(self.rfile.read(length).decode())
                path = urlparse(self.path).path
                if path == "/api/hold":
                    supervisor.request_hold()
                elif path == "/api/intervene":
                    supervisor.request_intervention()
                elif path == "/api/override":
                    supervisor.override_uncertain_checkpoint(
                        (params.get("note") or [""])[0]
                    )
                else:
                    self._json(404, {"error": "unknown endpoint"})
                    return
                self._json(200, supervisor.status())
            except Exception as error:
                self._json(409, {"error": f"{type(error).__name__}: {error}"})

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer((host, int(port)), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    return server, thread
