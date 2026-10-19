import io
import logging
from typing import List, Optional

import PIL.Image
from PIL import Image
import pypdfium2

logger = logging.getLogger(__name__)


def detect_file_type(header: bytes) -> Optional[str]:
    """
    Detect file format from magic bytes.
    - JPEG: FF D8 FF
    - PNG: 89 50 4E 47
    - WEBP: RIFF (bytes 0-3) + WEBP (bytes 8-11)
    - PDF: 25 50 44 46 (%PDF)
    """
    if header.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if header.startswith(b"\x89\x50\x4e\x47") or header.startswith(b"\x89PNG"):
        return "png"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "webp"
    if header.startswith(b"\x25\x50\x44\x46") or header.startswith(b"%PDF"):
        return "pdf"
    return None


def ingest(file_bytes: bytes, filename: str) -> List[PIL.Image.Image]:
    """
    Converts raw file bytes into a list of RGB PIL Images.

    - Detects file type from magic bytes (JPEG, PNG, WEBP, PDF).
    - PDF: Renders pages using pypdfium2 at scale 1.5 (caps at 5 pages).
           Falls back to pdf2image at dpi=150 if pypdfium2 fails.
    - Images: Decodes using PIL. If RGBA, composites over a white background with alpha mask.
              Converts any other non-RGB modes to RGB.
    - Raises ValueError for unsupported types or if no pages could be extracted.
    """
    if not file_bytes:
        raise ValueError(f"Cannot ingest empty file: '{filename}'")

    file_type = detect_file_type(file_bytes[:512])
    if not file_type:
        raise ValueError(
            f"Unsupported file type for '{filename}'. Magic bytes do not match JPEG, PNG, WEBP, or PDF."
        )

    images: List[PIL.Image.Image] = []

    if file_type == "pdf":
        try:
            with pypdfium2.PdfDocument(file_bytes) as doc:
                page_count = len(doc)
                if page_count == 0:
                    raise ValueError("PDF document has 0 pages")
                num_pages = min(page_count, 5)
                for i in range(num_pages):
                    page = doc.get_page(i)
                    bitmap = page.render(scale=1.5)
                    pil_img = bitmap.to_pil()
                    if pil_img.mode != "RGB":
                        pil_img = pil_img.convert("RGB")
                    images.append(pil_img)
        except Exception as e:
            logger.warning(
                f"pypdfium2 failed to process PDF '{filename}': {e}. Attempting fallback with pdf2image."
            )
            try:
                from pdf2image import convert_from_bytes

                rendered = convert_from_bytes(file_bytes, dpi=150, first_page=1, last_page=5)
                for img in rendered[:5]:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    images.append(img)
            except Exception as fallback_err:
                raise ValueError(
                    f"Failed to process PDF '{filename}' using pypdfium2 and fallback pdf2image: {fallback_err}"
                ) from fallback_err

    else:
        # Handle JPEG, PNG, WEBP
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img.load()

            if img.mode == "RGBA":
                # Composite onto white background using alpha mask
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            images.append(img)
        except Exception as img_err:
            raise ValueError(f"Failed to decode image '{filename}': {img_err}") from img_err

    if not images:
        raise ValueError(f"No pages could be extracted from document '{filename}'.")

    return images
