from typing import Optional
from pydantic import BaseModel


class UploadResponse(BaseModel):
    job_id: str
    status: str
    reference_id: Optional[str] = None
    poll_url: str
    message: str
