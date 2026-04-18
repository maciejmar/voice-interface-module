# Air-gapped alternative:
# FROM repo.bank.com.pl/docker/nvidia/cuda:12.4.1-runtime-ubuntu22.04
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    curl \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root && rm -rf /tmp/poetry_cache

# Piper TTS binary
RUN wget -q https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz \
    -O /tmp/piper.tar.gz \
    && tar -xzf /tmp/piper.tar.gz -C /tmp \
    && mv /tmp/piper/piper /usr/local/bin/piper \
    && chmod +x /usr/local/bin/piper \
    && rm -rf /tmp/piper.tar.gz /tmp/piper

# Polish TTS models (skip in air-gapped — mount as volume instead)
# These are downloaded here for convenience on internet-connected builds.
# In air-gapped: comment out the RUN block below and use volume mount only.
RUN mkdir -p /models/tts && \
    wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/darkman/medium/pl_PL-darkman-medium.onnx" \
        -O /models/tts/pl_PL-darkman-medium.onnx && \
    wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/darkman/medium/pl_PL-darkman-medium.onnx.json" \
        -O /models/tts/pl_PL-darkman-medium.onnx.json && \
    wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx" \
        -O /models/tts/pl_PL-gosia-medium.onnx && \
    wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx.json" \
        -O /models/tts/pl_PL-gosia-medium.onnx.json

RUN mkdir -p /models/stt

COPY app/main.py .

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8100/health || exit 1

CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "1"]
