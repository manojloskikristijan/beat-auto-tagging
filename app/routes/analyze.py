import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl

from app.services.audio_analysis import analyze_audio
from app.utils.downloader import download_audio, DownloadError

logger = logging.getLogger(__name__)

router = APIRouter()


class AnalyzeRequest(BaseModel):
    file_url: HttpUrl


class AnalyzeResponse(BaseModel):
    bpm: int
    key: str
    confidence: float


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeRequest) -> AnalyzeResponse:
    url = str(body.file_url)
    logger.info("Analyze request for %s", url)

    # Download the remote audio file to a local temp path
    try:
        audio_path = await download_audio(url)
    except DownloadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Run the CPU-bound analysis
    try:
        result = analyze_audio(audio_path)
    except Exception as exc:
        logger.exception("Analysis failed for %s", url)
        raise HTTPException(status_code=422, detail=f"Audio analysis failed: {exc}") from exc

    logger.info("Result for %s: %s", url, result)
    return AnalyzeResponse(**result)
