"""FastAPI application for DriveDocs. Route handlers are thin; business logic lives in drivedocs.service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from drivedocs.config import settings
from drivedocs.ingestion import (
    DuplicateDocumentError,
    FileSizeLimitExceededError,
    IngestionError,
    MalformedDocumentError,
    StorageLimitError,
    UnsupportedFileTypeError,
)
from drivedocs.models import ModelLoadError
from drivedocs.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    ErrorResponse,
    HealthResponse,
    PublicImageSearchResult,
    ReadinessResponse,
    SearchTextRequest,
    SystemInfoResponse,
    TextSearchResult,
    UploadResponse,
)
from drivedocs.service import (
    DriveDocsService,
    InvalidQueryImageError,
    NotFoundError,
    ServiceUnavailableError,
)

logger = logging.getLogger("drivedocs.api")
WEB_DIR = Path(__file__).resolve().parent / "web"
MULTIPART_OVERHEAD = 256 * 1024


def _configure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("drivedocs").setLevel(settings.log_level)


class APIError(Exception):
    def __init__(self, status_code: int, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.extra = extra


def _error(status_code: int, code: str, message: str, headers: Optional[Dict[str, str]] = None, **extra: Any):
    body = {"detail": {"code": code, "message": message, **jsonable_encoder(extra)}}
    return JSONResponse(status_code=status_code, content=body, headers=headers)


class _BodyTooLarge(StarletteHTTPException):
    """Raised while streaming an oversized body. It subclasses HTTPException so FastAPI's body parsing re-raises it
    unchanged (other exceptions there become a generic 400)."""

    def __init__(self) -> None:
        super().__init__(
            status_code=413,
            detail=f"The request body exceeds the maximum upload size of {settings.max_upload_size_bytes} bytes.",
        )


class BodySizeLimitMiddleware:
    """Reject request bodies larger than `max_bytes`, using Content-Length and by counting streamed bytes."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    if int(value) > self.max_bytes:
                        await self._reject(send)
                        return
                except ValueError:
                    pass
        received = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge()
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if not started:
                await self._reject(send)

    async def _reject(self, send: Send) -> None:
        response = _error(
            413,
            "file_too_large",
            f"The request body exceeds the maximum upload size of {settings.max_upload_size_bytes} bytes.",
        )
        await send({"type": "http.response.start", "status": response.status_code, "headers": response.raw_headers})
        await send({"type": "http.response.body", "body": response.body})


def create_service() -> DriveDocsService:
    return DriveDocsService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    service = create_service()
    app.state.service = service
    logger.info(
        "Starting DriveDocs %s (device=%s, storage=%s, image_search=%s, provider=%s)",
        settings.app_version,
        settings.device,
        service.storage.mode,
        settings.enable_image_search,
        service.generator.provider if service.generator.llm_configured else "extractive",
    )
    # Initialize in the background so the port binds immediately; /ready reports progress.
    service.start(background=True)
    yield
    logger.info("DriveDocs shutting down")


app = FastAPI(
    title="DriveDocs AI",
    version=settings.app_version,
    description=(
        "AI document intelligence API: ingest PDF/Markdown/text/images, search text and diagrams, and get answers "
        "grounded in retrieved evidence with verified citations. **No authentication**: do not upload private data "
        "to a public instance."
    ),
    lifespan=lifespan,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)

app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_upload_size_bytes + MULTIPART_OVERHEAD)
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

UI_CSP = (
    "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; connect-src 'self'; "
    "font-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("X-Frame-Options", "DENY")
    path = request.url.path
    if path == "/" or path.startswith("/assets"):
        response.headers.setdefault("Content-Security-Policy", UI_CSP)
    return response


# Error handling ----------------------------------------------------------------------------------------------------


@app.exception_handler(APIError)
async def _api_error(_: Request, exc: APIError):
    return _error(exc.status_code, exc.code, exc.message, **exc.extra)


@app.exception_handler(IngestionError)
async def _ingestion_error(_: Request, exc: IngestionError):
    if isinstance(exc, DuplicateDocumentError):
        return _error(409, exc.code, str(exc), existing_document=exc.existing_doc.model_dump())
    status_map = {
        UnsupportedFileTypeError: 400,
        FileSizeLimitExceededError: 413,
        MalformedDocumentError: 422,
        StorageLimitError: 507,
    }
    code = next((v for k, v in status_map.items() if isinstance(exc, k)), 400)
    return _error(code, exc.code, str(exc))


@app.exception_handler(ServiceUnavailableError)
async def _unavailable(_: Request, exc: ServiceUnavailableError):
    return _error(503, "service_unavailable", str(exc), headers={"Retry-After": "10"})


@app.exception_handler(ModelLoadError)
async def _model_error(_: Request, exc: ModelLoadError):
    return _error(503, "model_unavailable", "An embedding model could not be loaded. Retry later.")


@app.exception_handler(NotFoundError)
async def _not_found(_: Request, exc: NotFoundError):
    return _error(404, "not_found", str(exc))


@app.exception_handler(InvalidQueryImageError)
async def _bad_image(_: Request, exc: InvalidQueryImageError):
    return _error(422, "invalid_image", str(exc))


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError):
    errors = [
        {"field": ".".join(str(p) for p in err.get("loc", []) if p != "body"), "message": err.get("msg", "invalid")}
        for err in exc.errors()
    ]
    return _error(422, "validation_error", "The request is invalid.", errors=errors)


@app.exception_handler(StarletteHTTPException)
async def _http_error(_: Request, exc: StarletteHTTPException):
    code = {404: "not_found", 405: "method_not_allowed", 413: "file_too_large"}.get(exc.status_code, "http_error")
    message = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return _error(exc.status_code, code, message)


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception):
    logger.exception("Unhandled error: %s", type(exc).__name__)
    return _error(500, "internal_error", "An internal error occurred.")


# Dependencies ------------------------------------------------------------------------------------------------------


def get_service(request: Request) -> DriveDocsService:
    return request.app.state.service


def ready_service(request: Request) -> DriveDocsService:
    service: DriveDocsService = request.app.state.service
    service.require_ready()
    return service


async def _read_upload(upload: UploadFile, limit: int) -> bytes:
    buf = bytearray()
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > limit:
            raise FileSizeLimitExceededError(f"The file exceeds the maximum allowed limit of {limit} bytes.")
    return bytes(buf)


# Routes: status ----------------------------------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["status"], responses={503: {"model": HealthResponse}})
def health(service: DriveDocsService = Depends(get_service)):
    """Liveness check. Cheap: never loads models or touches indexes on disk.

    Returns 200 while starting and when ready; 503 only if start-up failed, so a broken deploy fails its health check.
    """
    report = service.health()
    return JSONResponse(status_code=503 if service.phase == "failed" else 200, content=report.model_dump())


@app.get("/ready", response_model=ReadinessResponse, tags=["status"], responses={503: {"model": ReadinessResponse}})
def ready(service: DriveDocsService = Depends(get_service)):
    """Readiness check: storage, registry, indexes and the text model. Returns 503 until ready."""
    report = service.readiness()
    return JSONResponse(status_code=200 if report.ready else 503, content=report.model_dump())


@app.get("/system/info", response_model=SystemInfoResponse, tags=["status"])
def system_info(service: DriveDocsService = Depends(get_service)) -> SystemInfoResponse:
    """Public configuration: limits, storage mode, enabled features and thresholds."""
    return service.system_info()


@app.get("/evaluation", tags=["evaluation"])
def evaluation(service: DriveDocsService = Depends(get_service)) -> Dict[str, Any]:
    """The latest evaluation report produced by scripts/run_eval.py (never computed on request)."""
    report = service.evaluation_report()
    if report is None:
        raise APIError(404, "evaluation_not_found", "No evaluation report has been generated yet.")
    return report


# Routes: documents -------------------------------------------------------------------------------------------------


@app.post("/documents/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED, tags=["documents"])
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
) -> UploadResponse:
    """Upload and ingest a document (PDF, TXT, MD, PNG, JPG). Duplicate content returns 409."""
    service: DriveDocsService = request.app.state.service
    await run_in_threadpool(service.require_ready)
    content = await _read_upload(file, settings.max_upload_size_bytes)
    return await run_in_threadpool(service.ingest, file.filename or "uploaded_document", content)


@app.get("/documents", response_model=DocumentListResponse, tags=["documents"])
def list_documents(service: DriveDocsService = Depends(ready_service)) -> DocumentListResponse:
    """List documents, including failed ones (status=failed with an error message)."""
    return service.list_documents()


@app.get("/documents/{doc_id}", response_model=DocumentDetailResponse, tags=["documents"])
def document_details(doc_id: str, service: DriveDocsService = Depends(ready_service)) -> DocumentDetailResponse:
    """Document metadata with its text chunks (page-referenced) and images."""
    return service.document_details(doc_id)


@app.delete("/documents/{doc_id}", response_model=DeleteResponse, tags=["documents"])
def delete_document(doc_id: str, service: DriveDocsService = Depends(ready_service)) -> DeleteResponse:
    """Delete a document, its index entries, its images and its stored original."""
    return service.delete_document(doc_id)


@app.get("/static/images/{image_name}", tags=["documents"], include_in_schema=False)
def get_image(image_name: str, service: DriveDocsService = Depends(get_service)):
    path = service.image_file(image_name)
    if path is None:
        raise APIError(404, "not_found", "Image not found.")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


# Routes: search & chat ---------------------------------------------------------------------------------------------


@app.post("/search/text", response_model=List[TextSearchResult], tags=["search"])
def search_text(req: SearchTextRequest, service: DriveDocsService = Depends(ready_service)) -> List[TextSearchResult]:
    """Semantic search over text chunks (MiniLM cosine similarity). `accepted` marks scores above the threshold."""
    return service.search_text(req.query, req.top_k)


@app.post("/search/images", response_model=List[PublicImageSearchResult], tags=["search"])
async def search_images(
    request: Request,
    query: Optional[str] = Form(None, max_length=500),
    top_k: int = Form(2, ge=1, le=10),
    image_file: Optional[UploadFile] = File(None),
) -> List[PublicImageSearchResult]:
    """Image search with CLIP: a text description (text→image) or an uploaded image (image→image)."""
    service: DriveDocsService = request.app.state.service
    await run_in_threadpool(service.require_ready)
    if image_file is not None and (image_file.filename or image_file.size):
        content = await _read_upload(image_file, settings.max_upload_size_bytes)
        if not content:
            raise APIError(422, "invalid_image", "The query image is empty.")
        return await run_in_threadpool(service.search_images_by_image, content, top_k)
    if query and query.strip():
        return await run_in_threadpool(service.search_images_by_text, query.strip(), top_k)
    raise APIError(400, "missing_query", "Provide either a text 'query' or an 'image_file' for image retrieval.")


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(req: ChatRequest, service: DriveDocsService = Depends(ready_service)) -> ChatResponse:
    """Retrieve text and visual evidence and return a grounded answer with verified citations."""
    return service.chat(req)


# Routes: web UI ----------------------------------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def serve_dashboard():
    return FileResponse(WEB_DIR / "index.html", media_type="text/html")


app.mount("/assets", StaticFiles(directory=str(WEB_DIR / "assets")), name="assets")
