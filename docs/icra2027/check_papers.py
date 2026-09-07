#!/usr/bin/env python3
"""Validate built paper PDFs, figure provenance, and numeric consistency."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run(*args):
    return subprocess.check_output(args, text=True, cwd=HERE)


def main():
    figure = json.loads((HERE / "figure_provenance.json").read_text())
    for relative, digest in figure["sources"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
    evidence = json.loads((HERE / "evidence.json").read_text())
    frames = evidence["cap"]["frames"]
    xyz = np.array([f["ee_xyz_m"] for f in frames])
    assert abs((xyz[1, 2]-xyz[0, 2])*1000 - 9.1748) < .001
    assert abs(np.linalg.norm(xyz[2]-xyz[1])*1000 - 107.4155) < .001
    assert abs(figure["orientation_deg"]["T7"]) < 1e-5
    assert abs(figure["orientation_deg"]["T6"] - 2.39) < .01
    results = {}
    for stem, pdf in [("paper_a", "learning_to_open_and_remove.pdf"), ("paper_b", "roboevolve.pdf")]:
        info = run("pdfinfo", pdf)
        count = int(re.search(r"Pages:\s+(\d+)", info)[1])
        assert 6 <= count <= 8, (pdf, count)
        assert "612 x 792" in info
        fonts = run("pdffonts", pdf)
        assert "Type 3" not in fonts
        for line in fonts.splitlines()[2:]:
            assert line.split()[-5] == "yes", line  # embedded flag
        text = run("pdftotext", "-layout", pdf, "-")
        assert "Anonymous Authors" in text and "OpenAI Codex assisted" in text
        for forbidden in ["yunaoikawa", "Oikawa", "/home/admin", "password", "Peacock", "Pasteur"]:
            assert forbidden not in text, (pdf, forbidden)
        assert "??" not in text
        log = (HERE / "build" / f"{stem}.log").read_text()
        assert "Overfull" not in log and "undefined" not in log
        results[pdf] = {"pages": count, "words_approx": len(text.split()),
                        "sha256": hashlib.sha256((HERE/pdf).read_bytes()).hexdigest(),
                        "font_embedding": "passed", "anonymous_text_check": "passed",
                        "references_and_layout": "passed"}
    (HERE / "build/validation.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
