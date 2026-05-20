"""统一异常处理 + 请求 ID 注入"""
import uuid
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from src.infrastructure.kb_manager import (
    KnowledgeBaseNotFoundError,
    DocumentNotFoundError,
    DuplicateNamespaceError,
)
from src.core.config import settings

logger = logging.getLogger(__name__)

def register_error_handlers(app: FastAPI):
    # 请求 ID 中间件
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        # ✅ 同步到 ContextVar，供 SSE 流式输出使用
        from src.api.deps import set_request_id
        set_request_id(request_id)

        # ✅ 新增：同步到日志系统的 trace_id
        from src.infrastructure.logger import set_trace_id
        set_trace_id(request_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        # ✅ 新增：请求结束后清除上下文
        from src.infrastructure.logger import clear_context
        clear_context()

        return response

    def _respond(request: Request, status: int, error: str, detail: str = ""):
        rid = getattr(request.state, "request_id", "N/A")
        logger.warning(f"[{rid}] {error}: {detail}")
        return JSONResponse(
            status_code=status,
            content={
                "error": error,
                "detail": detail,
                "request_id": rid,
            },
        )

    @app.exception_handler(KnowledgeBaseNotFoundError)
    async def kb_not_found(request, exc):
        return _respond(request, 404, "Knowledge base not found", str(exc))

    @app.exception_handler(DocumentNotFoundError)
    async def doc_not_found(request, exc):
        return _respond(request, 404, "Document not found", str(exc))

    @app.exception_handler(DuplicateNamespaceError)
    async def dup_ns(request, exc):
        return _respond(request, 409, "Duplicate namespace", str(exc))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        msg = exc.errors()[0]["msg"] if exc.errors() else "Invalid input"
        return _respond(request, 422, "Validation error", msg)

    @app.exception_handler(HTTPException)
    async def http_exc(request, exc):
        return _respond(request, exc.status_code, exc.detail or "HTTP error")

    @app.exception_handler(Exception)
    async def global_handler(request, exc):
        rid = getattr(request.state, "request_id", "N/A")
        logger.error(f"[{rid}] Unhandled error", exc_info=True)
        detail = str(exc) if settings.env == "dev" else "Internal server error"
        return _respond(request, 500, "Internal server error", detail)
