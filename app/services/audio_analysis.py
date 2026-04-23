"""High-level audio analysis: BPM + musical key via Essentia."""

import logging
from pathlib import Path

import essentia.standard as es

logger = logging.getLogger(__name__)

TARGET_SR: int = 44_100
KEY_PROFILE: str = "bgate"
BPM_METHOD: str = "multifeature"

_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def analyze_audio(file_path: Path) -> dict:
    """Analyze an audio file and return BPM, key, and key confidence."""
    logger.info("Loading audio from %s (sr=%d)", file_path, TARGET_SR)
    audio = es.MonoLoader(filename=str(file_path), sampleRate=TARGET_SR)()

    bpm_raw, _ticks, _rhythm_conf, _estimates, _intervals = es.RhythmExtractor2013(method=BPM_METHOD)(audio)
    bpm = int(float(bpm_raw) + 0.5)
    logger.info("Detected BPM: %s", bpm)

    key_letter, scale, strength = es.KeyExtractor(profileType=KEY_PROFILE)(audio)
    key_letter = _FLAT_TO_SHARP.get(key_letter, key_letter)
    key_string = f"{key_letter} {scale}"
    confidence = round(float(strength), 4)
    logger.info("Detected key: %s (confidence=%.4f)", key_string, confidence)

    return {"bpm": bpm, "key": key_string, "confidence": confidence}
