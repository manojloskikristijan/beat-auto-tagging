# Audio Analysis Service

A stateless FastAPI microservice that extracts musical features from audio files. Given a URL to an audio file, it returns the **BPM**, **musical key**, **confidence score**, and **genre / mood / instrument tags**.

---

## Features

- **BPM Detection** — Estimates tempo using Essentia's `RhythmExtractor2013` (multifeature method), a MIREX-tuned multi-estimator beat tracker.
- **Musical Key Detection** — Identifies the key (e.g. "A minor", "G major") using Essentia's `KeyExtractor` with the `bgate` profile (Faraldo 2016) — the production-grade default for modern pop and electronic music.
- **Confidence Score** — Reports how strongly the detected key stands out from alternatives (0–1).
- **Genre + Tags** — Top-3 Discogs genres (400-class taxonomy) plus top-5 mood/theme and instrument tags from the MTG-Jamendo models, all built on the shared Discogs-EffNet embedding.
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
  "bpm": 120,
  "key": "C major",
  "confidence": 0.7523,
  "genres": [
    { "label": "Electronic---House", "prob": 0.71 },
    { "label": "Electronic---Techno", "prob": 0.12 }
  ],
  "moods": [
    { "label": "energetic", "prob": 0.42 },
    { "label": "happy", "prob": 0.31 }
  ],
  "instruments": [
    { "label": "synthesizer", "prob": 0.61 },
    { "label": "drummachine", "prob": 0.55 }
  ]
}
```

| Field         | Type                             | Description                                                       |
|---------------|----------------------------------|-------------------------------------------------------------------|
| `bpm`         | int                              | Estimated tempo in beats per minute                               |
| `key`         | string                           | Detected musical key (e.g. "A minor")                             |
| `confidence`  | float                            | Key detection confidence between 0 and 1                          |
| `genres`      | array of `{label, prob}`         | Top-3 Discogs genres (400-class taxonomy), sorted by probability  |
| `moods`       | array of `{label, prob}`         | Top mood/theme tags from MTG-Jamendo (prob ≥ 0.10, max 5)         |
| `instruments` | array of `{label, prob}`         | Top instrument tags from MTG-Jamendo (prob ≥ 0.10, max 5)         |

Tag arrays may be empty if no class scores above the 0.10 confidence threshold.

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
│   │   ├── audio_analysis.py      # BPM + key detection via Essentia
│   │   └── classifier.py          # Genre + mood + instrument tags via Discogs-EffNet
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
| Audio Analysis  | Essentia (BPM, key) |
| Classification  | Essentia + TensorFlow, Discogs-EffNet + MTG-Jamendo heads |
| HTTP Client     | httpx (async)       |
| Containerization| Docker (python:3.10-slim) |

---

## Models

The Docker build downloads four pre-trained models (~150 MB total) from the [MTG model zoo](https://essentia.upf.edu/models/) into `/app/models/`:

| File                                                | Purpose                          |
|-----------------------------------------------------|----------------------------------|
| `discogs-effnet-bs64-1.pb`                          | Shared audio embedding (16 kHz)  |
| `genre_discogs400-discogs-effnet-1.pb`              | Genre head (400 Discogs classes) |
| `mtg_jamendo_moodtheme-discogs-effnet-1.pb`         | Mood/theme tag head              |
| `mtg_jamendo_instrument-discogs-effnet-1.pb`        | Instrument tag head              |

Models are loaded into memory once at process startup and reused across requests. Expect built image size ~1.5 GB and per-request CPU latency of ~5–10 s for a 3-minute track.
