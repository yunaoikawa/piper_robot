import json

import numpy as np
import plotly.graph_objects as go

from src.optimize_plotly_mobile import optimize


def test_mobile_optimizer_preserves_page_but_reduces_mesh(tmp_path):
    count = 3000
    x = np.linspace(0.0, 1.0, count + 2)
    figure = go.Figure(
        go.Mesh3d(
            x=x,
            y=np.sin(x),
            z=np.cos(x),
            i=np.arange(count),
            j=np.arange(1, count + 1),
            k=np.arange(2, count + 2),
        )
    )
    source = tmp_path / "full.html"
    output = tmp_path / "mobile" / "view.html"
    figure.write_html(source, include_plotlyjs=True)

    report = optimize(
        source,
        output,
        maximum_faces=120,
        maximum_points=100,
    )

    assert output.exists()
    assert (output.parent / "plotly.min.js").exists()
    assert report["original_faces"] == count
    assert report["displayed_faces"] == 120
    assert report["mobile_bytes"] < report["original_bytes"]
    page = output.read_text()
    assert 'src="plotly.min.js"' in page
    assert "Plotly.newPlot" in page
