import asyncio
import logging
from typing import Any, Dict, Optional

import httpx

from app.config import get_settings
from app.queue.job_store import job_store

logger = logging.getLogger(__name__)


async def _deliver_payload(job_id: str, webhook_url: str, payload: Dict[str, Any]) -> bool:
    """
    Internal helper to deliver a webhook payload with exponential backoff retry logic.
    - Uses httpx.AsyncClient with timeout from settings.WEBHOOK_TIMEOUT_SECONDS
    - Retries up to settings.WEBHOOK_RETRY_COUNT times
    - Backoff: sleep 2^attempt seconds between retries (1s, 2s, 4s...)
    - On success (2xx): updates job_store with webhook_delivered=True and returns True
    - On exhaustion: marks job as webhook_failed in job_store and returns False
    """
    if not webhook_url:
        logger.warning(f"No webhook URL provided for job {job_id}. Skipping webhook delivery.")
        return False

    settings = get_settings()
    max_retries = max(1, settings.WEBHOOK_RETRY_COUNT)
    timeout = settings.WEBHOOK_TIMEOUT_SECONDS

    for attempt in range(max_retries):
        try:
            logger.info(
                f"Delivering webhook for job_id={job_id} to {webhook_url} (attempt {attempt + 1}/{max_retries})"
            )
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(webhook_url, json=payload)
                if 200 <= response.status_code < 300:
                    logger.info(
                        f"Webhook delivery succeeded for job_id={job_id} on attempt {attempt + 1} (HTTP {response.status_code})"
                    )
                    job_store.set_webhook_delivered(job_id, True)
                    return True
                else:
                    logger.warning(
                        f"Webhook attempt {attempt + 1} for job_id={job_id} failed with HTTP {response.status_code}: {response.text[:200]}"
                    )
        except Exception as exc:
            logger.warning(
                f"Webhook attempt {attempt + 1} for job_id={job_id} encountered an error: {exc}"
            )

        # Backoff: 2^attempt seconds before retrying
        if attempt < max_retries - 1:
            backoff = 2**attempt
            logger.info(f"Backing off for {backoff}s before retry attempt {attempt + 2} for job_id={job_id}")
            await asyncio.sleep(backoff)

    logger.error(f"All {max_retries} webhook delivery attempts failed for job_id={job_id}")
    job_store.set_webhook_failed(job_id)
    return False


async def deliver_webhook(job_id: str, webhook_url: str, result: Dict[str, Any]) -> bool:
    """
    Deliver successful completion result to the customer's webhook endpoint.
    Payload: {"job_id": job_id, "status": "completed", "result": result}
    """
    payload = {
        "job_id": job_id,
        "status": "completed",
        "result": result,
    }
    if isinstance(result, dict) and result.get("reference_id"):
        payload["reference_id"] = result["reference_id"]

    return await _deliver_payload(job_id, webhook_url, payload)


async def deliver_webhook_error(job_id: str, webhook_url: str, error: str) -> bool:
    """
    Deliver error notification to the customer's webhook endpoint.
    Payload: {"job_id": job_id, "status": "failed", "error": error, "result": None}
    """
    job = job_store.get(job_id)
    payload = {
        "job_id": job_id,
        "status": "failed",
        "error": error,
        "result": None,
    }
    if job and job.reference_id:
        payload["reference_id"] = job.reference_id

    return await _deliver_payload(job_id, webhook_url, payload)
