#!/usr/bin/env python3
"""SAM 3.1 image segmentation server for the left observer camera.

The server deliberately keeps the wire protocol model-agnostic.  It accepts
JPEG frames and a concept prompt, and returns PNG masks.  Video-side tracking
and prompt refresh can be added without changing the pasteur client.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import zmq

from rollout.sam_segmentation import (
    MaskCandidate,
    decode_request,
    encode_error,
    encode_response,
)


class Sam3Backend:
    def __init__(self, checkpoint: str | None = None):
        try:
            import torch
            from PIL import Image
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:  # pragma: no cover - remote-only dependency
            raise RuntimeError(
                "SAM 3.1 server requires the sam3 checkout, torch, and Pillow"
            ) from exc
        self.torch = torch
        self.Image = Image
        kwargs = {} if checkpoint is None else {"checkpoint_path": checkpoint}
        self.model = build_sam3_image_model(**kwargs).eval()
        self.processor = Sam3Processor(self.model)

    def segment(self, image_bgr, prompt: str, confidence_threshold: float):
        rgb = image_bgr[:, :, ::-1]
        state = self.processor.set_image(self.Image.fromarray(rgb))
        output = self.processor.set_text_prompt(state=state, prompt=prompt)
        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")
        if masks is None or boxes is None or scores is None:
            return []
        masks = masks.detach().cpu().numpy()
        boxes = boxes.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy().reshape(-1)
        if masks.ndim == 4:
            masks = masks[:, 0]
        candidates = []
        for mask, box, score in zip(masks, boxes, scores):
            if float(score) < confidence_threshold:
                continue
            candidates.append(
                MaskCandidate(
                    mask=np.asarray(mask, dtype=bool),
                    box_xyxy=np.asarray(box, dtype=float),
                    score=float(score),
                )
            )
        return candidates


def serve(endpoint: str, checkpoint: str | None, prompt: str):
    backend = Sam3Backend(checkpoint)
    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(endpoint)
    print(f"SAM3 server listening on {endpoint}", flush=True)
    while True:
        parts = socket.recv_multipart()
        metadata = {}
        try:
            metadata, image = decode_request(parts)
            started = time.perf_counter()
            candidates = backend.segment(
                image,
                str(metadata.get("prompt") or prompt),
                float(metadata.get("confidence_threshold", 0.25)),
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            socket.send_multipart(
                encode_response(
                    frame_id=int(metadata["frame_id"]),
                    source_timestamp=float(metadata["timestamp"]),
                    model="sam3.1",
                    inference_ms=elapsed_ms,
                    candidates=candidates,
                )
            )
        except Exception as exc:  # keep REP socket usable after one bad frame
            frame_id = -1
            try:
                frame_id = int(metadata.get("frame_id", -1))
            except Exception:
                pass
            socket.send_multipart(encode_error(frame_id, str(exc)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="tcp://*:5562")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default="transparent circular petri-dish lid")
    args = parser.parse_args()
    serve(args.endpoint, args.checkpoint, args.prompt)


if __name__ == "__main__":
    main()
