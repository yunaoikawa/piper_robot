from __future__ import annotations

import json
import sys
import tempfile
import threading
from datetime import datetime
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.daily_scene import DailySceneStore, SceneObject
from src.daily_scene_ui import handler_for


def request(url, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    with urlopen(
        Request(
            url,
            data=body,
            headers={"content-type": "application/json"},
        )
    ) as response:
        return response.status, json.loads(response.read())


with tempfile.TemporaryDirectory() as directory:
    store = DailySceneStore(Path(directory) / "daily.json")
    now = datetime.now().astimezone().timestamp()
    draft = store.propose(
        objects=[
            SceneObject(
                "dish",
                "petri dish",
                {"type": "cylinder"},
                status="confirmed",
            )
        ],
        calibration_id="cal",
        camera_ids={"head": "phone"},
        timestamp_s=now,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(store))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert request(base + "/api/scene")[1]["revision"] == draft.revision
        confirmed = request(
            base + "/api/confirm",
            {"revision": draft.revision, "operator": "phone"},
        )[1]
        assert confirmed["status"] == "confirmed"
        changed = request(
            base + "/api/changed", {"reason": "moved dish"}
        )[1]
        assert changed["status"] == "change_reported"
        assert changed["revision"] == draft.revision + 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

print("daily scene UI checks passed")
