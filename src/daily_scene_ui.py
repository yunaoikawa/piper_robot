#!/usr/bin/env python3
"""Phone-friendly daily bench confirmation UI and JSON API."""

from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.daily_scene import DailySceneStore, SceneNotConfirmed, SceneObject


HTML = """<!doctype html><html lang="ja"><meta name="viewport"
content="width=device-width,initial-scale=1"><meta charset="utf-8">
<title>机上環境の確認</title><style>
body{font-family:-apple-system,sans-serif;margin:16px;background:#111;color:#eee}
button,input{font-size:18px;padding:12px;margin:5px;border-radius:10px}
button{background:#2677ff;color:white;border:0}.danger{background:#d33}
.card{background:#222;padding:12px;margin:10px 0;border-radius:12px}
img{width:100%;max-height:44vh;object-fit:contain;background:#333}
.uncertain{border:2px solid #f5b942}.confirmed{border:2px solid #42d17b}
</style><h1>今日の机上環境</h1><div id="status"></div><div id="images"></div>
<div id="objects"></div><button id="confirm">この状態で正しい</button>
<button id="add">物体を追加</button>
<button class="danger" id="changed">机上を変更した</button>
<script>
let scene=null;
async function api(path,body){let r=await fetch(path,{method:body?'POST':'GET',
headers:{'content-type':'application/json'},body:body?JSON.stringify(body):null});
let j=await r.json();if(!r.ok)throw Error(j.error||r.statusText);return j}
async function load(){scene=await api('/api/scene');let s=document.querySelector('#status');
if(!scene){s.textContent='スキャン結果がまだありません';return}
s.textContent=`状態: ${scene.status} / revision ${scene.revision}`;
document.querySelector('#images').innerHTML=Object.entries(scene.images||{}).map(
([k,v])=>`<div class=card><b>${k}</b><img src="/artifact?path=${encodeURIComponent(v)}"></div>`).join('');
document.querySelector('#objects').innerHTML=(scene.objects||[]).map((o,i)=>
`<div class="card ${o.status}"><input value="${o.semantic_name}" id="n${i}">
<span>${o.source} ${(o.confidence*100).toFixed(0)}%</span>
<div>形状 <select id="t${i}"><option value="box" ${((o.geometry||{}).kind||(o.geometry||{}).type)==='box'?'selected':''}>box</option>
<option value="cylinder" ${((o.geometry||{}).kind||(o.geometry||{}).type)==='cylinder'?'selected':''}>cylinder</option></select>
寸法(m) ${[0,1,2].map(j=>`<input style="width:72px" type="number" step="0.001"
id="s${i}_${j}" value="${((o.geometry||{}).size_xyz_m||[0,0,0])[j]||0}">`).join('')}</div>
<div>支持面 <input value="${o.role||''}" id="r${i}"></div>
<button onclick="setobj(${i},'confirmed')">確認</button>
<button onclick="setobj(${i},'absent')">ない</button></div>`).join('')}
async function setobj(i,status){scene.objects[i].semantic_name=document.querySelector('#n'+i).value;
scene.objects[i].role=document.querySelector('#r'+i).value||null;
let g=scene.objects[i].geometry||{};g.kind=document.querySelector('#t'+i).value;
g.size_xyz_m=[0,1,2].map(j=>Number(document.querySelector(`#s${i}_${j}`).value));
scene.objects[i].geometry=g;
scene.objects[i].status=status;scene=await api('/api/objects',{revision:scene.revision,objects:scene.objects});load()}
document.querySelector('#add').onclick=async()=>{let name=prompt('追加する物体名');
if(!name)return;scene.objects.push({instance_id:'operator-'+Date.now(),semantic_name:name,
geometry:{type:'measured_static_surface'},confidence:1,status:'confirmed',
source:'operator',transparent:false,depth_quality:'operator_identity'});
scene=await api('/api/objects',{revision:scene.revision,objects:scene.objects});load()};
document.querySelector('#confirm').onclick=async()=>{try{await api('/api/confirm',
{revision:scene.revision,operator:'mobile-ui'});load()}catch(e){alert(e.message)}};
document.querySelector('#changed').onclick=async()=>{let reason=prompt('何を変更しましたか？')||'operator_reported_change';
await api('/api/changed',{reason});load()};load();setInterval(load,5000);
</script></html>"""


def handler_for(store: DailySceneStore):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, value, status=HTTPStatus.OK):
            encoded = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _body(self):
            length = int(self.headers.get("content-length", "0"))
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                encoded = HTML.encode()
                self.send_response(200)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("content-length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            elif parsed.path == "/api/scene":
                scene = store.load()
                self._json(None if scene is None else scene.to_dict())
            elif parsed.path == "/artifact":
                requested = parse_qs(parsed.query).get("path", [""])[0]
                artifact = Path(requested)
                scene = store.load()
                allowed = set()
                if scene is not None:
                    allowed.update(str(Path(path).resolve()) for path in scene.images.values())
                    allowed.update(
                        str(Path(item.mask_path).resolve())
                        for item in scene.objects
                        if item.mask_path
                    )
                if (
                    not artifact.is_file()
                    or str(artifact.resolve()) not in allowed
                    or artifact.suffix.lower()
                    not in {".png", ".jpg", ".jpeg", ".webp"}
                ):
                    return self._json({"error": "image not found"}, 404)
                encoded = artifact.read_bytes()
                content_type = (
                    "image/png"
                    if artifact.suffix.lower() == ".png"
                    else "image/jpeg"
                )
                self.send_response(200)
                self.send_header("content-type", content_type)
                self.send_header("content-length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            try:
                body = self._body()
                if self.path == "/api/confirm":
                    scene = store.confirm(
                        revision=body["revision"],
                        operator=body.get("operator", "mobile-ui"),
                    )
                elif self.path == "/api/changed":
                    scene = store.report_change(body.get("reason", "operator change"))
                elif self.path == "/api/objects":
                    scene = store.replace_objects(
                        [SceneObject.from_dict(item) for item in body["objects"]],
                        revision=body["revision"],
                    )
                else:
                    return self._json({"error": "not found"}, 404)
                self._json(scene.to_dict())
            except (KeyError, ValueError, SceneNotConfirmed) as error:
                self._json({"error": str(error)}, 409)

        def log_message(self, format, *args):
            pass

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="/tmp/piper_daily_scene.json")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer(
        (args.host, args.port), handler_for(DailySceneStore(args.scene))
    )
    print(f"http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
