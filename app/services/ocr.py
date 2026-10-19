from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import PIL.Image
from PIL import Image

logger = logging.getLogger(__name__)

# Ensure PaddlePaddle oneDNN flags are set for CPU inference stability
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["PADDLE_PDX_EAGER_INIT"] = "0"

# Make PaddleX initialization idempotent to prevent 'PDX has already been initialized' crash
try:
    import paddlex.repo_manager.core as pdx_core

    _orig_pdx_initialize = pdx_core.initialize

    def _safe_pdx_initialize(*args, **kwargs):
        if pdx_core._GlobalContext.is_initialized():
            return
        return _orig_pdx_initialize(*args, **kwargs)

    pdx_core.initialize = _safe_pdx_initialize
except Exception:
    pass


class OCREngine:
    """Singleton pattern manager for PaddleOCR engine."""

    _paddle_instance: Optional[Any] = None

    @classmethod
    def get_paddle(cls) -> Any:
        """Retrieve or initialize the PaddleOCR singleton instance."""
        if cls._paddle_instance is None:
            from paddleocr import PaddleOCR

            try:
                cls._paddle_instance = PaddleOCR(
                    use_angle_cls=True,
                    use_doc_unwarping=False,
                    use_doc_orientation_classify=False,
                    lang="en",
                    show_log=False,
                )
            except (ValueError, TypeError):
                cls._paddle_instance = PaddleOCR(
                    use_angle_cls=True,
                    use_doc_unwarping=False,
                    use_doc_orientation_classify=False,
                    lang="en",
                )
        return cls._paddle_instance


def _parse_paddle_raw(ocr_output: Any) -> List[Dict[str, Any]]:
    """Parse raw PaddleOCR output into structured block dictionaries."""
    blocks: List[Dict[str, Any]] = []

    if not ocr_output:
        return blocks

    # Handle PaddleX output format (list of dicts containing dt_polys, rec_texts, rec_scores)
    if isinstance(ocr_output, list) and len(ocr_output) > 0 and isinstance(ocr_output[0], dict):
        item = ocr_output[0]
        polys = item.get("dt_polys", [])
        texts = item.get("rec_texts", [])
        scores = item.get("rec_scores", [])

        for poly, text, score in zip(polys, texts, scores):
            pts = np.array(poly)
            x1 = int(np.min(pts[:, 0]))
            y1 = int(np.min(pts[:, 1]))
            x2 = int(np.max(pts[:, 0]))
            y2 = int(np.max(pts[:, 1]))
            clean_text = str(text).strip()
            if clean_text:
                blocks.append({
                    "bbox": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "text": clean_text,
                    "confidence": float(score),
                    "method": "PADDLE_CRNN",
                })
        return blocks

    # Handle classic PaddleOCR format: [[ [box, (text, score)], ... ]]
    lines = (
        ocr_output[0]
        if (
            isinstance(ocr_output, list)
            and len(ocr_output) > 0
            and isinstance(ocr_output[0], list)
        )
        else ocr_output
    )

    if isinstance(lines, list):
        for line in lines:
            if not line or len(line) < 2:
                continue
            box, rec = line[0], line[1]
            if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                text, score = rec[0], rec[1]
            else:
                text, score = str(rec), 1.0

            pts = np.array(box)
            x1 = int(np.min(pts[:, 0]))
            y1 = int(np.min(pts[:, 1]))
            x2 = int(np.max(pts[:, 0]))
            y2 = int(np.max(pts[:, 1]))
            clean_text = str(text).strip()
            if clean_text:
                blocks.append({
                    "bbox": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "text": clean_text,
                    "confidence": float(score),
                    "method": "PADDLE_CRNN",
                })

    return blocks


def run_ocr(pil_image: PIL.Image.Image) -> Dict[str, Any]:
    """
    Run PaddleOCR recognition on a document image.

    Returns:
        dict:
            blocks: List[dict] with bbox, text, confidence, and method
            raw_text: str (blocks joined with newlines)
            mean_confidence: float
            total_blocks: int
    """
    try:
        img_rgb = pil_image.convert("RGB")
        img_np = np.array(img_rgb)

        paddle_engine = OCREngine.get_paddle()

        # Run PaddleOCR inference
        if hasattr(paddle_engine, "predict"):
            raw_output = list(paddle_engine.predict(img_np))
        else:
            raw_output = paddle_engine.ocr(img_np, cls=True)

        blocks = _parse_paddle_raw(raw_output)

        # Sort all blocks top-to-bottom, left-to-right by bbox y1 coordinate
        blocks.sort(key=lambda b: (b["bbox"][0][1], b["bbox"][0][0]))

        # Build raw_text and calculate mean confidence
        raw_text = "\n".join(b["text"] for b in blocks)
        mean_confidence = (
            float(np.mean([b["confidence"] for b in blocks])) if blocks else 0.0
        )

        return {
            "blocks": blocks,
            "raw_text": raw_text,
            "mean_confidence": round(mean_confidence, 4),
            "total_blocks": len(blocks),
        }
    except Exception as exc:
        logger.error(f"PaddleOCR execution failed: {exc}. Returning empty OCR fallback.")
        return {
            "blocks": [],
            "raw_text": "",
            "mean_confidence": 0.0,
            "total_blocks": 0,
        }
