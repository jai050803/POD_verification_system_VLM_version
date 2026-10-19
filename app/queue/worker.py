import asyncio
import logging
from typing import Any, Dict

from app.config import get_settings
from app.queue.job_store import job_store

logger = logging.getLogger(__name__)

# Expose the asyncio queue with maxsize from settings
processing_queue: asyncio.Queue = asyncio.Queue(
    maxsize=get_settings().QUEUE_MAX_SIZE
)


async def process_job(item: Dict[str, Any]) -> None:
    """Process a single queued job item."""
    job_id = item.get("job_id")
    if not job_id:
        return
    logger.info(f"Worker picked up job {job_id}")
    try:
        job_store.set_processing(job_id)
        # Initial placeholder: will be wired to VLM service and webhook pipeline
        await asyncio.sleep(0.01)
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        job_store.set_failed(job_id, str(e))


async def start_worker() -> None:
    """Continuous background worker loop consuming tasks from processing_queue."""
    logger.info("Background processing worker started.")
    while True:
        try:
            item = await processing_queue.get()
            try:
                await process_job(item)
            finally:
                processing_queue.task_done()
        except asyncio.CancelledError:
            logger.info("Background processing worker cancelled. Stopping.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in worker loop: {e}")
            await asyncio.sleep(0.5)
