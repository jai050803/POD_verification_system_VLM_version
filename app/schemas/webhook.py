from typing import Optional
from pydantic import BaseModel
from app.schemas.result import PODResult


class WebhookPayload(BaseModel):
    job_id: str
    status: str
    reference_id: Optional[str] = None
    result: Optional[PODResult] = None
    error: Optional[str] = None
