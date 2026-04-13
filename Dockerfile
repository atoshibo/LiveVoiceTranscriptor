# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    ca-certificates \
    build-essential \
    cmake

COPY requirements.txt requirements.optional.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

ARG INSTALL_OPTIONAL_MODEL_DEPS=false
ARG PYTORCH_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_OPTIONAL_MODEL_DEPS" = "true" ]; then \
    PIP_EXTRA_INDEX_URL=$PYTORCH_EXTRA_INDEX_URL pip install -r requirements.optional.txt; \
    fi

COPY app/ app/
COPY scripts/ scripts/

RUN mkdir -p /data/sessions /data/certs

ENV SESSIONS_DIR=/data/sessions
ENV TLS_CERT_FILE=/data/certs/dev-cert.pem
ENV TLS_KEY_FILE=/data/certs/dev-key.pem
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV SERVER_PORT=8443
ENV TLS_ENABLED=true
ENV TLS_AUTO_GENERATE_SELF_SIGNED=true

FROM base AS server

EXPOSE 8443

CMD ["python", "-m", "app.main"]

FROM base AS worker

CMD ["python", "-m", "app.workers.worker"]
