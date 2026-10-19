import io
from typing import Optional
from fastapi import Depends, File, HTTPException, UploadFile, status

from app.config import Settings, get_settings


def detect_mime_type(header: bytes) -> Optional[str]:
    """
    Detect MIME type and file format from magic bytes.
    - JPEG: FF D8 FF
    - PNG: 89 50 4E 47
    - WEBP: first 4 bytes RIFF + bytes 8-11 WEBP
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


async def validate_upload_file(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> UploadFile:
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024

    # 1. Check Content-Length header if present
    if file.headers and "content-length" in file.headers:
        try:
            content_length = int(file.headers["content-length"])
            if content_length > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File size exceeds maximum allowed limit of {settings.MAX_FILE_SIZE_MB}MB",
                )
        except (ValueError, TypeError):
            pass

    # 2. Check actual file size on disk / spool
    try:
        file.file.seek(0, io.SEEK_END)
        actual_size = file.file.tell()
        file.file.seek(0)
        if actual_size > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File size ({actual_size / (1024 * 1024):.2f}MB) exceeds maximum allowed limit of {settings.MAX_FILE_SIZE_MB}MB",
            )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        pass

    # 3. Read first 512 bytes to inspect magic bytes
    header = await file.read(512)
    detected_type = detect_mime_type(header)
    await file.seek(0)

    if not detected_type:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported media type. Magic bytes do not match supported formats (JPEG, PNG, WEBP, PDF).",
        )

    # 4. Check against allowed extensions
    allowed = [ext.lower().lstrip(".") for ext in settings.ALLOWED_EXTENSIONS]
    type_matches = {
        "jpeg": ["jpg", "jpeg"],
        "png": ["png"],
        "webp": ["webp"],
        "pdf": ["pdf"],
    }
    allowed_for_type = type_matches.get(detected_type, [])
    if not any(ext in allowed for ext in allowed_for_type):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type '{detected_type}' is not allowed by configuration. Allowed types: {', '.join(allowed)}",
        )

    return file
