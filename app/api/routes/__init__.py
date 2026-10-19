from app.api.routes.health import router as health_router
from app.api.routes.upload import router as upload_router
from app.api.routes.jobs import router as jobs_router

__all__ = ["health_router", "upload_router", "jobs_router"]
