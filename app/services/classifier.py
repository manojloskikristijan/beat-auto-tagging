"""Genre + tag classification via Essentia's Discogs-EffNet models.

Models are loaded once at import time and reused across requests.
If the TensorFlow build of Essentia or the model files are unavailable
(e.g. local dev on Apple Silicon), the classifier is disabled and
classify() returns empty arrays so the rest of the service still works.
"""

import json
import logging
import os
from pathlib import Path

import essentia.standard as es
import numpy as np

logger = logging.getLogger(__name__)

CLASSIFIER_SR: int = 16_000
TOP_K: int = 5
TAG_THRESHOLD: float = 0.10

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/app/models"))

EMBEDDING_MODEL = MODELS_DIR / "discogs-effnet-bs64-1.pb"
GENRE_MODEL = MODELS_DIR / "genre_discogs400-discogs-effnet-1.pb"
GENRE_LABELS = MODELS_DIR / "genre_discogs400-discogs-effnet-1.json"
MOOD_MODEL = MODELS_DIR / "mtg_jamendo_moodtheme-discogs-effnet-1.pb"
MOOD_LABELS = MODELS_DIR / "mtg_jamendo_moodtheme-discogs-effnet-1.json"
INSTRUMENT_MODEL = MODELS_DIR / "mtg_jamendo_instrument-discogs-effnet-1.pb"
INSTRUMENT_LABELS = MODELS_DIR / "mtg_jamendo_instrument-discogs-effnet-1.json"


def _load_labels(metadata_path: Path) -> list[str]:
    with metadata_path.open() as f:
        return json.load(f)["classes"]


_embedding = None
_genre_head = None
_mood_head = None
_instrument_head = None
_genre_labels: list[str] = []
_mood_labels: list[str] = []
_instrument_labels: list[str] = []
_enabled = False

try:
    logger.info("Loading Discogs-EffNet models from %s", MODELS_DIR)
    _embedding = es.TensorflowPredictEffnetDiscogs(
        graphFilename=str(EMBEDDING_MODEL),
        output="PartitionedCall:1",
    )
    _genre_head = es.TensorflowPredict2D(
        graphFilename=str(GENRE_MODEL),
        input="serving_default_model_Placeholder",
        output="PartitionedCall:0",
    )
    _mood_head = es.TensorflowPredict2D(
        graphFilename=str(MOOD_MODEL),
        output="model/Sigmoid",
    )
    _instrument_head = es.TensorflowPredict2D(
        graphFilename=str(INSTRUMENT_MODEL),
        output="model/Sigmoid",
    )
    _genre_labels = _load_labels(GENRE_LABELS)
    _mood_labels = _load_labels(MOOD_LABELS)
    _instrument_labels = _load_labels(INSTRUMENT_LABELS)
    _enabled = True
    logger.info(
        "Classifier ready: genre=%d classes, mood=%d, instrument=%d",
        len(_genre_labels), len(_mood_labels), len(_instrument_labels),
    )
except (AttributeError, RuntimeError, FileNotFoundError, OSError) as exc:
    logger.warning(
        "Classifier disabled (%s: %s). "
        "Server will respond with empty genres/moods/instruments. "
        "Build inside Docker for full classification.",
        type(exc).__name__, exc,
    )


def _top_k(probs: np.ndarray, labels: list[str], k: int, threshold: float) -> list[dict]:
    """Return the top-k labels above `threshold`, sorted by probability descending."""
    idx = np.argsort(probs)[::-1][:k]
    return [
        {"label": labels[i], "prob": round(float(probs[i]), 4)}
        for i in idx
        if probs[i] >= threshold
    ]


def classify(file_path: Path) -> dict:
    """Run genre + mood + instrument classification on a single audio file."""
    if not _enabled:
        return {"genres": [], "moods": [], "instruments": []}

    logger.info("Classifying %s (sr=%d)", file_path, CLASSIFIER_SR)
    audio = es.MonoLoader(
        filename=str(file_path),
        sampleRate=CLASSIFIER_SR,
        resampleQuality=4,
    )()

    embeddings = _embedding(audio)

    genre_probs = _genre_head(embeddings).mean(axis=0)
    mood_probs = _mood_head(embeddings).mean(axis=0)
    instrument_probs = _instrument_head(embeddings).mean(axis=0)

    return {
        "genres": _top_k(genre_probs, _genre_labels, k=3, threshold=TAG_THRESHOLD),
        "moods": _top_k(mood_probs, _mood_labels, k=TOP_K, threshold=TAG_THRESHOLD),
        "instruments": _top_k(instrument_probs, _instrument_labels, k=TOP_K, threshold=TAG_THRESHOLD),
    }
