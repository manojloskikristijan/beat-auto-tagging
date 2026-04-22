"""Musical key detection using chroma features.

Strategy:
1. Normalize chroma per-frame, then take the median across time (robust to
   outlier frames like silence or transients).
2. Correlate the resulting 12-bin pitch profile against three sets of
   major/minor key templates rotated to every root note (24 candidates each).
3. Each profile set casts a weighted vote (its best correlation) for its
   winning key. The key with the highest total score wins.
4. Confidence reflects the margin between the top two keys.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# --- Key profile templates (C-rooted) ---

# Krumhansl-Kessler (1990) — perceptual probe-tone ratings
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Temperley (2007) "Music and Probability" — corpus-derived
TEMPERLEY_MAJOR = np.array([0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400])
TEMPERLEY_MINOR = np.array([0.712, 0.084, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330])

# Albrecht & Shanahan (2013) — large modern corpus, strong on pop/rock
ALBRECHT_MAJOR = np.array([0.238, 0.006, 0.111, 0.006, 0.137, 0.094, 0.016, 0.214, 0.009, 0.080, 0.008, 0.081])
ALBRECHT_MINOR = np.array([0.220, 0.006, 0.104, 0.123, 0.019, 0.103, 0.012, 0.214, 0.062, 0.022, 0.061, 0.052])

PROFILE_SETS = [
    (KK_MAJOR, KK_MINOR),
    (TEMPERLEY_MAJOR, TEMPERLEY_MINOR),
    (ALBRECHT_MAJOR, ALBRECHT_MINOR),
]

NOTE_NAMES: list[str] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _correlate(profile: npt.NDArray[np.floating], chroma_mean: npt.NDArray[np.floating]) -> float:
    """Pearson correlation between a key profile and the observed chroma vector."""
    return float(np.corrcoef(profile, chroma_mean)[0, 1])


def _score_all_keys(
    major: npt.NDArray[np.floating],
    minor: npt.NDArray[np.floating],
    chroma_vec: npt.NDArray[np.floating],
) -> list[tuple[str, float]]:
    """Score all 24 keys for one profile set. Returns [(key_name, corr), ...]."""
    results: list[tuple[str, float]] = []
    for shift in range(12):
        results.append((f"{NOTE_NAMES[shift]} major", _correlate(np.roll(major, shift), chroma_vec)))
        results.append((f"{NOTE_NAMES[shift]} minor", _correlate(np.roll(minor, shift), chroma_vec)))
    return results


def detect_key(chroma: npt.NDArray[np.floating]) -> dict[str, str | float]:
    """Return the detected key and confidence from a chroma feature matrix.

    Parameters
    ----------
    chroma : ndarray of shape (12, T)
        Chromagram (e.g. from librosa.feature.chroma_cqt).

    Returns
    -------
    dict with keys ``key`` (str, e.g. "A minor") and ``confidence`` (float 0-1).
    """
    # L1-normalize each frame so loud frames don't dominate, then take median
    # across time (more robust than mean to silence / transient outliers).
    norms = np.sum(chroma, axis=0, keepdims=True) + 1e-12
    chroma_normed = chroma / norms
    chroma_vec: npt.NDArray[np.floating] = np.median(chroma_normed, axis=1)

    # Accumulate weighted votes from every profile set.
    # Each set contributes its correlation score to its best key.
    key_scores: dict[str, float] = {}

    for major_prof, minor_prof in PROFILE_SETS:
        scored = _score_all_keys(major_prof, minor_prof, chroma_vec)
        # Sort descending by correlation
        scored.sort(key=lambda x: x[1], reverse=True)
        winner_name, winner_corr = scored[0]
        key_scores[winner_name] = key_scores.get(winner_name, 0.0) + winner_corr

    # The key with the highest accumulated score wins
    best_key = max(key_scores, key=key_scores.get)

    # Confidence: how far the winner is ahead of the runner-up.
    sorted_scores = sorted(key_scores.values(), reverse=True)
    if len(sorted_scores) >= 2:
        top = sorted_scores[0]
        runner = sorted_scores[1]
        # Normalize margin to 0-1 range
        confidence = (top - runner) / (abs(top) + 1e-8)
    else:
        confidence = 1.0

    confidence = round(max(0.0, min(1.0, confidence)), 4)

    return {"key": best_key, "confidence": confidence}
