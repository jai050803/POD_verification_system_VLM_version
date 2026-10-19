import asyncio
import logging
from typing import Any, Dict

from app.config import get_settings
from app.queue.job_store import job_store
from app.services.pipeline import run_pipeline
from app.services.webhook import deliver_webhook, deliver_webhook_error

logger = logging.getLogger(__name__)

# Module-level asyncio queue instance
_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(
    maxsize=get_settings().QUEUE_MAX_SIZE
)

# Alias for backwards compatibility with earlier prompts
processing_queue = _queue


async def enqueue(job_payload: Dict[str, Any]) -> None:
    """Put a job payload onto the processing queue."""
    _queue.put_nowait(job_payload)


def get_queue_size() -> int:
    """Return the current number of queued items."""
    return _queue.qsize()


async def worker_loop() -> None:
    """
    Continuous background worker loop consuming jobs from the asyncio queue.
    Executes the synchronous pipeline inside a thread executor to avoid blocking the event loop.
    Delivers results or errors to the specified webhook URL.
    """
    logger.info("Background worker loop started.")
    loop = asyncio.get_running_loop()

    while True:
        try:
            job = await _queue.get()
            job_id = job["job_id"]
            webhook_url = job["webhook_url"]
            reference_id = job.get("reference_id")
            doc_type_hint = job.get("doc_type_hint")
            filename = job.get("filename", f"file_{job_id}")

            logger.info(f"[{job_id}] State transition -> PROCESSING")
            job_store.set_processing(job_id)

            try:
                # Run the synchronous CPU/network pipeline in thread pool
                result = await loop.run_in_executor(
                    None,
                    run_pipeline,
                    job_id,
                    job["file_bytes"],
                    filename,
                    webhook_url,
                    reference_id,
                    doc_type_hint,
                )

                logger.info(f"[{job_id}] State transition -> COMPLETED")
                job_store.set_completed(job_id, result)

                # Deliver webhook
                logger.info(f"[{job_id}] Delivering success webhook to {webhook_url}")
                await deliver_webhook(job_id, webhook_url, result)

            except Exception as exc:
                err_msg = str(exc)
                logger.error(f"[{job_id}] State transition -> FAILED. Error: {err_msg}", exc_info=True)
                job_store.set_failed(job_id, err_msg)

                # Deliver error webhook
                logger.info(f"[{job_id}] Delivering error webhook to {webhook_url}")
                try:
                    await deliver_webhook_error(job_id, webhook_url, err_msg)
                except Exception as wh_err:
                    logger.error(f"[{job_id}] Failed to deliver error webhook: {wh_err}")

            finally:
                _queue.task_done()

        except asyncio.CancelledError:
            logger.info("Worker loop received cancellation. Shutting down worker.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in worker loop: {e}", exc_info=True)
            await asyncio.sleep(0.5)


# Alias for main.py lifespan compatibility
start_worker = worker_loop
