from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.routes.analyze import router as analyze_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="Audio Analysis Service",
    version="1.0.0",
    description="Stateless microservice that extracts BPM, musical key, and confidence from audio files.",
)

app.include_router(analyze_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.exception_handler(Exception)
async def global_exception_handler(_request, exc: Exception) -> JSONResponse:
    logging.getLogger("app").exception("Unhandled error")
    return JSONResponse(status_code=500, content={"error": "Internal server error"})
