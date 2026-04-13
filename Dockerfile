FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    ca-certificates \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.optional.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ARG INSTALL_OPTIONAL_MODEL_DEPS=false
RUN if [ "$INSTALL_OPTIONAL_MODEL_DEPS" = "true" ]; then \
    pip install --no-cache-dir -r requirements.optional.txt; \
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
