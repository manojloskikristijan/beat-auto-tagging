"""High-level audio analysis: BPM + key detection."""

import logging
from pathlib import Path

import librosa
import numpy as np

from app.services.key_detection import detect_key

logger = logging.getLogger(__name__)

# Analyze at most the first 60 seconds to keep latency low
MAX_DURATION_SECONDS: float = 60.0
# Target sample rate — 22 050 Hz is librosa's default and sufficient for
# tempo / chroma analysis while keeping memory usage reasonable.
TARGET_SR: int = 22_050


def analyze_audio(file_path: Path) -> dict:
    """Analyze an audio file and return BPM, key, and confidence.

    Parameters
    ----------
    file_path : Path
        Local path to the downloaded audio file.

    Returns
    -------
    dict with ``bpm`` (float), ``key`` (str), and ``confidence`` (float).
    """
    logger.info("Loading audio from %s (first %.0fs, sr=%d)", file_path, MAX_DURATION_SECONDS, TARGET_SR)

    y, sr = librosa.load(
        file_path,
        sr=TARGET_SR,
        mono=True,
        duration=MAX_DURATION_SECONDS,
    )

    # Separate harmonic and percussive components once, reuse for both tasks.
    # Percussive → cleaner beat signal for BPM.
    # Harmonic → cleaner pitch signal for key detection.
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # --- BPM ---
    # Use percussive component with smaller hop_length (256) for finer
    # temporal resolution (~11.6 ms per frame at 22 050 Hz).
    tempo, _ = librosa.beat.beat_track(y=y_percussive, sr=sr, hop_length=256)
    # librosa >= 0.10 returns an ndarray; squeeze to scalar
    bpm = float(np.atleast_1d(tempo).flat[0])
    bpm = round(bpm, 2)
    logger.info("Detected BPM: %s", bpm)

    # --- Key ---
    # chroma_cqt uses constant-Q transform — logarithmically spaced bins that
    # align with musical intervals, producing more accurate pitch profiles
    # than the linear-spaced chroma_stft.
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, n_chroma=12)
    key_result = detect_key(chroma)
    logger.info("Detected key: %s (confidence=%.4f)", key_result["key"], key_result["confidence"])

    return {
        "bpm": bpm,
        "key": key_result["key"],
        "confidence": key_result["confidence"],
    }
