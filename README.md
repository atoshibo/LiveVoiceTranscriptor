# LiveVoiceTranscriptor

Canonical pipeline server for continuous multilingual audio capture, transcription, enrichment, and memory retrieval.

## Architecture

The server implements the canonical pipeline specification:
- **30-second transport chunks** at ingest (from app)
- **30-second decode windows** with **15-second stride** (server-side)
- **15-second stabilization stripes** as the commit unit
- Multi-ASR execution with bounded LLM reconciliation
- Canonical stabilized segments as the single source of truth

### Pipeline Stages (12)

1. `normalize_audio` - Ingest and normalize to 16kHz mono
2. `first_pass_medium` - Quick transcription with faster-whisper:medium
3. `speaker_diarization` - Selective diarization (when justified)
4. `acoustic_triage` - Classify speech vs non-speech regions
5. `decode_lattice` - Build 30s/15s overlapping decode windows
6. `candidate_asr_large_v3` - faster-whisper:large-v3 on all windows
7. `candidate_asr_parakeet` - nemo-asr:parakeet-tdt-0.6b-v3 on all windows
8. `stripe_grouping` - Group evidence by 15s stabilization stripes
9. `reconciliation` - Bounded LLM arbitration with deterministic fallback
10. `canonical_assembly` - Build canonical stabilized segments
11. `selective_enrichment` - Speaker labels where justified
12. `derived_outputs` - SRT, VTT, quality report, etc.

### Default Models

| Role | Model | Notes |
|------|-------|-------|
| First pass | faster-whisper:medium | Quick initial transcription |
| Candidate A | faster-whisper:large-v3 | High-quality multilingual |
| Candidate B | nemo-asr:parakeet-tdt-0.6b-v3 | **Replaces Whisper Turbo** (English-only, honest coverage) |
| Reconciler | Local LLM (Qwen 7B GGUF) | Bounded local arbitration, deterministic fallback |

## Quick Start

### Prerequisites

- Python 3.10+
- Redis server
- GPU recommended (CUDA)

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and edit environment config
cp .env.example .env
# Edit .env with your model paths and auth token

# 3. Start Redis (if not running)
# Windows: download from https://github.com/microsoftarchive/redis/releases
# Linux/Mac: redis-server

# 4. Start the server
# Windows:
scripts\start_server.bat
# Linux/Mac:
bash scripts/start_server.sh

# 5. Start the worker (separate terminal)
# Windows:
scripts\start_worker.bat
# Linux/Mac:
bash scripts/start_worker.sh

# 6. Open browser
# http://localhost:8000
```

### Docker

```bash
# Set your models directory
export MODELS_DIR=/path/to/your/models

# Start everything
docker-compose up -d

# Open browser
# http://localhost:8000
```

### Running Tests

```bash
# Windows:
scripts\run_tests.bat

# Linux/Mac:
PYTHONPATH=. python -m pytest app/tests/ -v
```

## API Compatibility

This server is **fully backward-compatible** with the v0.4.2 API that the Android/device app uses. All endpoints, request schemas, response fields, auth headers, and alias behaviors are preserved.

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v2/health` | GET | Health check (no auth) |
| `/api/v2/sessions` | POST | Create session |
| `/api/v2/sessions/{id}/chunks` | POST | Upload audio chunk |
| `/api/v2/sessions/{id}/finalize` | POST | Start processing |
| `/api/v2/jobs/{id}` | GET | Job status |
| `/api/v2/sessions/{id}/transcript` | GET | Final transcript |
| `/api/v2/sessions/{id}/transcript/partial` | GET | Partial transcript |
| `/api/v2/sessions/{id}/transcript/speaker` | GET | Speaker transcript |
| `/api/v2/sessions/{id}/subtitle.srt` | GET | SRT subtitles |
| `/api/v2/sessions/{id}/subtitle.vtt` | GET | VTT subtitles |
| `/api/v2/models` | GET | Model registry |
| `/api/v2/sessions/grouped` | GET | Grouped sessions |

### Auth

- `Authorization: Bearer <token>`
- `X-Api-Token: <token>`

## Model Paths

Models are resolved from `MODELS_DIR` with support for both friendly names and HuggingFace cache-style directories:

```
models/
  parakeet-tdt-0.6b-v3/         # Friendly name
  models--nvidia--parakeet-tdt-0.6b-v3/  # HF cache style
  canary-1b-v2/
  whisper-large-v3-turbo-hf/
  models--Systran--faster-whisper-medium/
  models--Systran--faster-whisper-large-v3/
  pyannote-speaker-diarization-community-1/
  Qwen2.5-7B-Instruct-Q5_K_M.gguf
```

## Storage Layout

```
sessions/{session_id}/
  chunks/        - Raw uploaded transport chunks
  raw/           - Original merged audio + ingest metadata
  normalized/    - Session-normalized audio timeline
  triage/        - Acoustic tags, speech islands
  windows/       - Synthesized decode windows
  candidates/    - Per-model raw ASR outputs
  reconciliation/ - Stripe packets, arbitration outputs
  canonical/     - Canonical segments + final transcript
  derived/       - Subtitles, quality, classification
  current/       - Compatibility read surface (API reads here)
  pipeline/      - Pipeline run tracking
```
