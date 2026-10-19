from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    WEBHOOK_FAILED = "WEBHOOK_FAILED"


class Job(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    webhook_url: str
    reference_id: Optional[str] = None
    doc_type_hint: Optional[str] = None  # dashboard can hint: POD, DL, RC, BILTI
    created_at: float
    updated_at: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    pages_processed: int = 0
    webhook_delivered: bool = False
