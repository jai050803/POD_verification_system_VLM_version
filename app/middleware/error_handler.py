import logging
import traceback
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.schemas.error import ErrorResponse

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            return await call_next(request)
        except Exception as exc:
            request_id = getattr(request.state, "request_id", None)
            logger.error(
                f"Unhandled exception during request [request_id={request_id}]: {exc}\n{traceback.format_exc()}"
            )
            error_response = ErrorResponse(
                error="Internal Server Error",
                detail="An unexpected error occurred while processing your request.",
                request_id=request_id,
            )
            headers = {"X-Request-ID": request_id} if request_id else None
            return JSONResponse(
                status_code=500,
                content=error_response.model_dump(),
                headers=headers,
            )
