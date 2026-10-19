import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.api.routes import health_router, jobs_router, upload_router
from app.middleware.error_handler import ErrorHandlerMiddleware
from app.middleware.request_id import RequestIDMiddleware
from app.queue.worker import processing_queue, start_worker

# Expose the asyncio queue as required
queue = processing_queue


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle manager."""
    # Launch background worker task on startup
    worker_task = asyncio.create_task(start_worker())
    yield
    # Graceful shutdown of worker task
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="POD Processing Backend",
    version="1.0.0",
    description="Asynchronous document processing backend for Proof of Delivery (POD) verification.",
    lifespan=lifespan,
)

# Middleware stack
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(RequestIDMiddleware)

# Routers: health at /, others under /api/v1
app.include_router(health_router)
app.include_router(health_router, prefix="/api/v1")
app.include_router(upload_router, prefix="/api/v1")
app.include_router(jobs_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
