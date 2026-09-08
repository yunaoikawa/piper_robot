#!/usr/bin/env python3
"""Serve a phone-friendly tap-and-confirm target UI; never moves the robot."""

from __future__ import annotations

import argparse
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rollout.operator_target_selection import (  # noqa: E402
    validate_target_selection,
    write_target_selection,
)


PAGE = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>{title}</title><style>
html,body{{margin:0;background:#111;color:#fff;font-family:-apple-system,sans-serif}}
main{{max-width:900px;margin:auto;padding-bottom:calc(18px + env(safe-area-inset-bottom))}}
h1,p{{margin:12px 16px}}#wrap{{position:relative;touch-action:manipulation}}
#target{{display:block;width:100%;height:auto}}#mark{{position:absolute;width:34px;height:34px;
border:4px solid #ffeb00;border-radius:50%;transform:translate(-50%,-50%);display:none;
box-shadow:0 0 0 2px #000}}#mark:before,#mark:after{{content:'';position:absolute;background:#ffeb00}}
#mark:before{{left:13px;top:-9px;width:4px;height:44px}}#mark:after{{left:-9px;top:13px;width:44px;height:4px}}
.buttons{{display:flex;gap:10px;padding:14px 16px}}button{{flex:1;font-size:17px;font-weight:700;
padding:14px;border:0;border-radius:12px}}#confirm{{background:#ffd900;color:#111}}#reset{{background:#444;color:#fff}}
#status{{padding:0 16px;color:#ddd}}.ok{{color:#69f08b!important}}
</style></head><body><main><h1>{title}</h1>
<p>対象の中心を画像上でタップし、黄色い印を確認してから確定してください。確定前はロボットを動かしません。</p>
<div id="wrap"><img id="target" src="/target.jpg?v={version}"><div id="mark"></div></div>
<div class="buttons"><button id="reset">やり直す</button><button id="confirm" disabled>この点を中心として確定</button></div>
<div id="status">未選択</div></main><script>
const img=document.getElementById('target'),mark=document.getElementById('mark');
const status=document.getElementById('status'),confirmButton=document.getElementById('confirm');let chosen=null;
function setPoint(u,v,confirmed=false){{chosen={{u,v}};mark.style.left=(100*u/img.naturalWidth)+'%';
mark.style.top=(100*v/img.naturalHeight)+'%';mark.style.display='block';confirmButton.disabled=false;
status.textContent=`選択: (${{u.toFixed(1)}}, ${{v.toFixed(1)}})`+(confirmed?' — 確定済み':' — 未確定');
status.className=confirmed?'ok':'';}}
img.addEventListener('click',e=>{{const r=img.getBoundingClientRect();
setPoint((e.clientX-r.left)*img.naturalWidth/r.width,(e.clientY-r.top)*img.naturalHeight/r.height);}});
document.getElementById('reset').onclick=()=>{{chosen=null;mark.style.display='none';confirmButton.disabled=true;
status.textContent='未選択';status.className='';}};
confirmButton.onclick=async()=>{{if(!chosen)return;confirmButton.disabled=true;
const response=await fetch('/selection',{{method:'POST',headers:{{'Content-Type':'application/json'}},
body:JSON.stringify({{...chosen,confirmed:true}})}});const result=await response.json();
if(!response.ok){{status.textContent='エラー: '+result.error;confirmButton.disabled=false;return;}}
setPoint(result.pixel_uv[0],result.pixel_uv[1],true);}};
img.addEventListener('load',async()=>{{try{{const r=await fetch('/selection');if(r.ok){{const s=await r.json();
setPoint(s.pixel_uv[0],s.pixel_uv[1],true);}}}}catch(e){{}}}});
</script></body></html>"""


class TargetHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, image: Path, selection: Path, semantic_name: str, title: str, **kwargs):
        self.image = image
        self.selection = selection
        self.semantic_name = semantic_name
        self.title = title
        super().__init__(*args, **kwargs)

    def _send(self, body: bytes, content_type: str, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?", 1)[0] in {"/", "/current.html"}:
            body = PAGE.format(title=self.title, version=self.image.stat().st_mtime_ns).encode()
            return self._send(body, "text/html; charset=utf-8")
        if self.path.split("?", 1)[0] == "/target.jpg":
            return self._send(self.image.read_bytes(), "image/jpeg")
        if self.path.split("?", 1)[0] == "/selection" and self.selection.exists():
            return self._send(self.selection.read_bytes(), "application/json")
        return self._send(b'{"error":"not found"}', "application/json", HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if self.path != "/selection":
            return self._send(b'{"error":"not found"}', "application/json", HTTPStatus.NOT_FOUND)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            image = cv2.imread(str(self.image))
            if image is None:
                raise ValueError("target image is unreadable")
            selection = validate_target_selection(
                payload,
                semantic_name=self.semantic_name,
                image_path=self.image,
                image_width_px=image.shape[1],
                image_height_px=image.shape[0],
            )
            write_target_selection(self.selection, selection)
            body = json.dumps(selection.to_dict()).encode()
            return self._send(body, "application/json")
        except (ValueError, json.JSONDecodeError) as error:
            body = json.dumps({"error": str(error)}).encode()
            return self._send(body, "application/json", HTTPStatus.BAD_REQUEST)

    def log_message(self, message, *args):
        sys.stderr.write("[target-ui] " + message % args + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--semantic-name", required=True)
    parser.add_argument("--title", default="対象中心を選択")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8771)
    args = parser.parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    handler = partial(
        TargetHandler,
        image=args.image.resolve(),
        selection=args.selection.resolve(),
        semantic_name=args.semantic_name,
        title=args.title,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"target selection UI: http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
