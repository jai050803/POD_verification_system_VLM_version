from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import PIL.Image
from PIL import Image

logger = logging.getLogger(__name__)

# Global cached orientation classifier instance to avoid reloading weights
_ORIENTATION_CLASSIFIER: Optional[Any] = None


def get_orientation_classifier() -> Optional[Any]:
    """Lazy-load and cache the PaddleOCR document orientation classifier."""
    global _ORIENTATION_CLASSIFIER
    if _ORIENTATION_CLASSIFIER is None:
        try:
            from paddleocr import DocImgOrientationClassification

            _ORIENTATION_CLASSIFIER = DocImgOrientationClassification()
        except Exception:
            try:
                from paddleocr import PaddleOCR

                _ORIENTATION_CLASSIFIER = PaddleOCR(use_doc_orientation_classify=True)
            except Exception:
                _ORIENTATION_CLASSIFIER = None
    return _ORIENTATION_CLASSIFIER


def correct_rotation(image: Image.Image) -> Tuple[Image.Image, bool, int]:
    """
    Step 0 — Rotation Correction:
    - Use PaddleOCR's angle classifier to detect document rotation.
    - If rotation is detected (90, 180, 270 degrees), correct it using
      PIL Image.rotate() with expand=True.
    - If PaddleOCR classifier is unavailable or fails, returns original image.
    """
    try:
        classifier = get_orientation_classifier()
        if classifier is None:
            return image, False, 0

        # Downscale a copy for orientation detection to prevent heavy multi-megapixel latency
        w, h = image.size
        max_side = max(w, h)
        if max_side > 1000:
            scale = 1000.0 / max_side
            thumb = image.resize(
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                resample=getattr(Image, "Resampling", Image).BILINEAR,
            )
        else:
            thumb = image

        img_np = np.array(thumb.convert("RGB"))

        results = list(classifier.predict(img_np))
        if not results:
            return image, False, 0

        pred = results[0]
        label = None
        if isinstance(pred, dict):
            if "label_names" in pred and pred["label_names"]:
                label = pred["label_names"][0]
            elif "label" in pred:
                label = pred["label"]
        elif hasattr(pred, "label_names") and pred.label_names:
            label = pred.label_names[0]

        if label is not None:
            angle = int(label)
            if angle in (90, 180, 270):
                image = image.rotate(angle, expand=True)
                return image, True, angle
    except Exception as e:
        logger.debug(f"Rotation correction skipped due to classifier error: {e}")

    return image, False, 0


def cap_resolution(image: Image.Image, max_edge: int = 1800) -> Image.Image:
    """
    Step 1 — Resolution Cap:
    - If the longest edge of the image exceeds max_edge (1800px),
      resize proportionally to fit within 1800px.
    - Uses PIL Image.LANCZOS for high quality resampling.
    """
    width, height = image.size
    longest_edge = max(width, height)

    if longest_edge > max_edge:
        scale = float(max_edge) / float(longest_edge)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resample_filter = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize((new_width, new_height), resample=resample_filter)

    return image


def preprocess(
    image_input: Union[str, Path, Image.Image, np.ndarray, Any],
    return_meta: bool = True,
) -> Union[PIL.Image.Image, Tuple[PIL.Image.Image, Dict[str, Any]]]:
    """
    Preprocess a document image for VLM:
    1. Ensures RGB color representation.
    2. Rotation correction using PaddleOCR angle classifier (if available).
    3. Resolution cap to 1800px longest edge using LANCZOS.

    Note: Binarization (adaptive thresholding) and contrast filters (CLAHE, gray-world)
    are intentionally omitted because modern VLMs (e.g. Gemini 2.0 / Qwen2.5-VL) excel
    on continuous-tone images, whereas binarization destroys nuanced visual artifacts.
    """
    if isinstance(image_input, (str, Path)):
        path_str = str(image_input).strip()
        if path_str.lower().endswith(".pdf"):
            raise ValueError(
                "PDF file input detected. Please convert PDF pages to images first before preprocessing."
            )
        if not os.path.isfile(path_str):
            raise FileNotFoundError(f"Image file not found: {path_str}")
        image = Image.open(path_str)
    elif isinstance(image_input, Image.Image):
        image = image_input
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input)
    else:
        raise ValueError(
            f"Unsupported input type '{type(image_input).__name__}'. Expected an image or file path."
        )

    # Ensure RGB representation
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Step 0 — Rotation Correction
    image, rotation_corrected, rotation_angle = correct_rotation(image)

    # Step 1 — Resolution Cap (max longest edge 1800px)
    image = cap_resolution(image, max_edge=1800)

    preprocess_meta = {
        "rotation_corrected": rotation_corrected,
        "rotation_angle": rotation_angle,
    }

    if return_meta:
        return image, preprocess_meta
    return image


def preprocess_image(pil_image: PIL.Image.Image) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
    """
    Primary backend service entry point:
    Preprocesses a PIL image with rotation detection and resolution capping.
    On any unexpected error, safely returns the original image with rotation_corrected=False.
    """
    try:
        processed_image, meta = preprocess(pil_image, return_meta=True)
        return processed_image, meta
    except Exception as exc:
        logger.warning(
            f"Image preprocessing encountered an error: {exc}. Falling back to original image."
        )
        return pil_image, {"rotation_corrected": False, "rotation_angle": 0}
