# syntax=docker/dockerfile:1
# Lightweight server image — no GPU deps, no model weights.
# GPU work is handled by the worker (see Dockerfile.worker).
FROM python:3.11-slim

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    ca-certificates

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

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

EXPOSE 8443

CMD ["python", "-m", "app.main"]
