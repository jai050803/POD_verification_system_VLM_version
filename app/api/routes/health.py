from typing import Any, Dict, List
from fastapi import APIRouter

from app.queue.job_store import job_store
from app.queue.models import Job
from app.queue.worker import processing_queue

router = APIRouter(tags=["Health & Admin"])


@router.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint returning system status, queue size, and total jobs count."""
    return {
        "status": "ok",
        "queue_size": processing_queue.qsize(),
        "jobs_total": len(job_store.list_all()),
    }


@router.get("/admin/jobs", response_model=List[Job])
async def admin_list_jobs() -> List[Job]:
    """Debug/admin endpoint listing all stored jobs."""
    return job_store.list_all()
