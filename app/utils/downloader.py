"""Download a remote audio file to a local temporary path."""

import logging
import tempfile
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

MAX_FILE_SIZE: int = 20 * 1024 * 1024  # 20 MB
ALLOWED_CONTENT_TYPES: set[str] = {
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mp3",
    "audio/flac",
    "audio/ogg",
    # Some CDNs serve audio as octet-stream
    "application/octet-stream",
    "binary/octet-stream",
}
DOWNLOAD_TIMEOUT: float = 30.0


class DownloadError(Exception):
    """Raised when the remote file cannot be fetched or is invalid."""


async def download_audio(url: str) -> Path:
    """Stream-download the file at *url* and return a local temp path.

    Raises ``DownloadError`` on network failures, size violations, or
    unsupported content types.
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=DOWNLOAD_TIMEOUT) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()

                # Validate content type (if provided)
                content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                if content_type and content_type not in ALLOWED_CONTENT_TYPES:
                    raise DownloadError(f"Unsupported content type: {content_type}")

                # Check declared size early
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > MAX_FILE_SIZE:
                    raise DownloadError(f"File too large ({int(content_length)} bytes, max {MAX_FILE_SIZE})")

                # Infer extension from URL or content type
                suffix = _infer_suffix(url, content_type)

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                downloaded = 0

                async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                    downloaded += len(chunk)
                    if downloaded > MAX_FILE_SIZE:
                        tmp.close()
                        Path(tmp.name).unlink(missing_ok=True)
                        raise DownloadError(f"File exceeds {MAX_FILE_SIZE} byte limit")
                    tmp.write(chunk)

                tmp.close()
                logger.info("Downloaded %d bytes to %s", downloaded, tmp.name)
                return Path(tmp.name)

    except httpx.HTTPStatusError as exc:
        raise DownloadError(f"HTTP {exc.response.status_code} fetching {url}") from exc
    except httpx.RequestError as exc:
        raise DownloadError(f"Failed to fetch {url}: {exc}") from exc


def _infer_suffix(url: str, content_type: str) -> str:
    """Best-effort file extension from URL path or content type."""
    url_path = url.split("?")[0].lower()
    for ext in (".mp3", ".wav", ".flac", ".ogg"):
        if url_path.endswith(ext):
            return ext

    ct_map = {"audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/wav": ".wav", "audio/x-wav": ".wav"}
    return ct_map.get(content_type, ".audio")
