from __future__ import annotations

import os
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router


# ---------------------------------------------------------------------------
# ✅ LOGGING (SAFE ADD)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title="Plagiarism Checker API",
        description=(
            "Multi-strategy plagiarism detection engine combining "
            "shingling, TF-IDF cosine similarity, and semantic embeddings."
        ),
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # --- CORS ---
    raw_origins = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000,null",
    )
    origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- API routes ---
    app.include_router(router)

    # --- Serve static frontend ---
    frontend_dir = Path(__file__).parent.parent / "frontend"

    if frontend_dir.exists():
        static_dir = frontend_dir / "assets"
        index_file = frontend_dir / "index.html"

        if static_dir.exists():
            app.mount(
                "/static",
                StaticFiles(directory=str(static_dir)),
                name="static",
            )
        else:
            logger.warning("⚠️ Static assets directory not found: %s", static_dir)

        if index_file.exists():
            @app.get("/", include_in_schema=False)
            async def serve_index(_: Request) -> FileResponse:
                return FileResponse(str(index_file))
        else:
            logger.warning("⚠️ index.html not found: %s", index_file)

    else:
        logger.warning("⚠️ Frontend directory not found")

    # --- Global exception handler ---
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return app


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------
app = create_app()


# ---------------------------------------------------------------------------
# ✅ OPTIONAL MODEL PRELOAD (SAFE CONTROL)
# ---------------------------------------------------------------------------
if os.getenv("PRELOAD_MODEL", "false").lower() == "true":

    @app.on_event("startup")
    async def load_model():
        try:
            from .core.similarity import _get_semantic_model
            _get_semantic_model()
            logger.info("✅ Semantic model preloaded")
        except Exception as e:
            logger.exception("❌ Failed to preload model: %s", e)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
