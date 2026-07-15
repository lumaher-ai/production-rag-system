from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from production_rag.exceptions import PaddingtonError
from production_rag.logging_config import get_logger

logger = get_logger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(PaddingtonError)
    async def production_rag_error_handler(request: Request, exc: PaddingtonError) -> JSONResponse:
        logger.warning(
            "domain_error",
            error_type=type(exc).__name__,
            detail=exc.detail,
            status_code=exc.status_code,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "error_type": type(exc).__name__},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        # Catch-all so the client always receives a structured JSON body instead
        # of a raw stack-trace 500. The full traceback is logged server-side.
        logger.error(
            "unhandled_error",
            error_type=type(exc).__name__,
            detail=str(exc),
            path=request.url.path,
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error_type": type(exc).__name__},
        )
