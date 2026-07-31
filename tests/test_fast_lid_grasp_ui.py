import json

import numpy as np
import pytest

from src.fast_lid_grasp_ui import TapSelectionStore, _page


def test_mobile_page_preserves_normalized_tap_and_requires_frame_identity():
    image = np.full((100, 200, 3), 80, np.uint8)
    store = TapSelectionStore(image, 12.0)
    page = _page(store).decode()
    assert "この蓋を掴む" in page
    assert "naturalWidth" in page
    selected = store.select(
        {"u": 0.25, "v": 0.75, "frame_sha256": store.frame_hash}
    )
    assert selected.uv == (0.25, 0.75)
    assert store.event.is_set()
    with pytest.raises(ValueError, match="画像が変わりました"):
        store.select({"u": 0.2, "v": 0.3, "frame_sha256": "0" * 64})
