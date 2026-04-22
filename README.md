# Audio Analysis Service

A stateless FastAPI microservice that extracts musical features from audio files. Given a URL to an audio file, it returns the **BPM**, **musical key**, and a **confidence score**.

---

## Features

- **BPM Detection** — Estimates tempo using percussive source separation and beat tracking via librosa.
- **Musical Key Detection** — Identifies the key (e.g. "A minor", "G major") using chroma analysis with an ensemble of three musicological key profiles:
  - Krumhansl-Kessler (1990)
  - Temperley (2007)
  - Albrecht & Shanahan (2013)
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
│   │   ├── audio_analysis.py      # Orchestrates BPM + key detection with librosa
│   │   └── key_detection.py       # Key detection using chroma correlation
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
- FFmpeg and libsndfile (system dependencies for audio processing)

### Local Setup

```bash
# Install system dependencies (macOS)
brew install ffmpeg libsndfile

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

MP3, WAV, FLAC, and OGG — up to 20 MB per file. Only the first 60 seconds of audio are analyzed to keep response times low.

---

## Tech Stack

| Component       | Technology          |
|-----------------|---------------------|
| Framework       | FastAPI + Uvicorn   |
| Audio Analysis  | librosa, NumPy, SciPy |
| HTTP Client     | httpx (async)       |
| Containerization| Docker (python:3.10-slim) |
