# Audio Analysis Service

A stateless FastAPI microservice that extracts musical features from audio files. Given a URL to an audio file, it returns the **BPM**, **musical key**, and a **confidence score**.

---

## Features

- **BPM Detection** — Estimates tempo using Essentia's `RhythmExtractor2013` (multifeature method), a MIREX-tuned multi-estimator beat tracker.
- **Musical Key Detection** — Identifies the key (e.g. "A minor", "G major") using Essentia's `KeyExtractor` with the `bgate` profile (Faraldo 2016) — the production-grade default for modern pop and electronic music.
- **Confidence Score** — Reports how strongly the detected key stands out from alternatives (0–1).
- **Streaming Download** — Fetches remote audio via streaming with size limits (20 MB) and content-type validation.
- **Containerized** — Ships with a production-ready Dockerfile.

---

## API

### `POST /analyze`

Analyze an audio file from a remote URL.

**Request body:**

```json
{
  "file_url": "https://example.com/track.mp3"
}
```

**Response:**

```json
{
  "bpm": 120.0,
  "key": "C major",
  "confidence": 0.7523
}
```

| Field        | Type    | Description                                |
|--------------|---------|--------------------------------------------|
| `bpm`        | float   | Estimated tempo in beats per minute        |
| `key`        | string  | Detected musical key (e.g. "A minor")      |
| `confidence` | float   | Key detection confidence between 0 and 1   |

### `GET /health`

Returns `{"status": "ok"}` — useful for load balancer and container health checks.

---

## Project Structure

```
audio_service/
├── app/
│   ├── main.py                    # FastAPI app setup, health endpoint, error handling
│   ├── routes/
│   │   └── analyze.py             # POST /analyze endpoint
│   ├── services/
│   │   └── audio_analysis.py      # BPM + key detection via Essentia
│   └── utils/
│       └── downloader.py          # Async audio file downloader with validation
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- FFmpeg (runtime dependency for Essentia's audio loader)

### Local Setup

```bash
# Install system dependencies (macOS)
brew install ffmpeg

# Install Python dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build the image
docker build -t audio-service .

# Run the container
docker run -p 8000:8000 audio-service
```

The service will be available at `http://localhost:8000`.

---

## Supported Audio Formats

MP3, WAV, FLAC, and OGG — up to 20 MB per file. The full track is analyzed.

---

## Tech Stack

| Component       | Technology          |
|-----------------|---------------------|
| Framework       | FastAPI + Uvicorn   |
| Audio Analysis  | Essentia, NumPy     |
| HTTP Client     | httpx (async)       |
| Containerization| Docker (python:3.10-slim) |
