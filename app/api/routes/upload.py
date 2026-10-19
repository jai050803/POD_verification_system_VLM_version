import asyncio
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status

from app.api.dependencies import validate_upload_file
from app.queue.job_store import job_store
from app.queue.worker import enqueue, processing_queue
from app.schemas.upload import UploadResponse

router = APIRouter(tags=["Upload"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload POD Document",
    description="Accepts a document file and queues it for background VLM processing.",
)
async def upload_document(
    file: UploadFile = Depends(validate_upload_file),
    webhook_url: str = Form(..., description="Webhook URL for delivering results"),
    reference_id: Optional[str] = Form(None, description="Optional tracking or reference ID"),
    doc_type_hint: Optional[str] = Form(None, description="Hint: POD, DL, RC, BILTI"),
) -> UploadResponse:
    job_id = str(uuid4())

    # Create job in job_store
    job = job_store.create(
        job_id=job_id,
        webhook_url=webhook_url,
        reference_id=reference_id,
        doc_type_hint=doc_type_hint,
    )

    # Read file bytes into memory
    file_bytes = await file.read()

    task_payload = {
        "job_id": job_id,
        "file_bytes": file_bytes,
        "filename": file.filename or f"upload_{job_id}",
        "webhook_url": webhook_url,
        "reference_id": reference_id,
        "doc_type_hint": doc_type_hint,
    }

    try:
        await enqueue(task_payload)
    except asyncio.QueueFull:
        job_store.set_failed(job_id, "Processing queue full, retry later")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Processing queue full, retry later",
        )

    return UploadResponse(
        job_id=job_id,
        status=job.status.value,
        reference_id=reference_id,
        poll_url=f"/api/v1/jobs/{job_id}",
        message="Document upload accepted and queued for processing",
    )
