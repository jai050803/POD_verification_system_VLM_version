from fastapi import APIRouter, HTTPException, status

from app.queue.job_store import job_store
from app.queue.models import Job

router = APIRouter(tags=["Jobs"])


@router.get(
    "/jobs/{job_id}",
    response_model=Job,
    summary="Get Job Status and Details",
    description="Fetches the status and results of a processing job by its job_id.",
)
async def get_job(job_id: str) -> Job:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    return job
