FROM python:3.10-slim

# FFmpeg is required at runtime by Essentia's AudioLoader (mp3/ogg/flac decoding).
# curl is used at build time to fetch the pre-trained models.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# essentia-tensorflow is a ~290 MB wheel; bump pip's timeout/retries so a slow
# or flaky connection doesn't abort the build mid-download.
RUN pip install --no-cache-dir --timeout 120 --retries 10 -r requirements.txt

# Pre-trained Discogs-EffNet models (shared embedding + 3 classification heads).
# Hosted by the MTG (Music Technology Group, UPF Barcelona).
ENV MODELS_DIR=/app/models
RUN mkdir -p ${MODELS_DIR} && \
    curl -fsSL -o ${MODELS_DIR}/discogs-effnet-bs64-1.pb \
        https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb && \
    curl -fsSL -o ${MODELS_DIR}/genre_discogs400-discogs-effnet-1.pb \
        https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb && \
    curl -fsSL -o ${MODELS_DIR}/genre_discogs400-discogs-effnet-1.json \
        https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json && \
    curl -fsSL -o ${MODELS_DIR}/mtg_jamendo_moodtheme-discogs-effnet-1.pb \
        https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb && \
    curl -fsSL -o ${MODELS_DIR}/mtg_jamendo_moodtheme-discogs-effnet-1.json \
        https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.json && \
    curl -fsSL -o ${MODELS_DIR}/mtg_jamendo_instrument-discogs-effnet-1.pb \
        https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.pb && \
    curl -fsSL -o ${MODELS_DIR}/mtg_jamendo_instrument-discogs-effnet-1.json \
        https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.json

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
