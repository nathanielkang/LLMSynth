#!/usr/bin/env bash
# Sync LLMSynth 2_Code pack to GCP VM (code only — no .env).
set -euo pipefail

PROJECT="${GCP_PROJECT:-knu-fog-experiments}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE="${GCP_INSTANCE:-llmsynth-gpu}"
LOCAL_PACK="${LOCAL_PACK:-$(cd "$(dirname "$0")/../.." && pwd)}"
REMOTE_DIR="${REMOTE_DIR:-~/llmsynth/2_Code}"

echo "[rsync] Project=$PROJECT Zone=$ZONE Instance=$INSTANCE"
echo "[rsync] Local=$LOCAL_PACK -> ${INSTANCE}:${REMOTE_DIR}"

gcloud compute scp --recurse \
  --project "$PROJECT" --zone "$ZONE" \
  "$LOCAL_PACK/" "${INSTANCE}:${REMOTE_DIR}/" \
  --exclude ".env" --exclude ".env.*" \
  --exclude "__pycache__" --exclude "results_paper3" \
  --exclude "results_smoke" --exclude "cache" \
  --exclude "*.pt" --exclude "*.pth"

echo "[rsync] Done. SSH: gcloud compute ssh --project $PROJECT --zone $ZONE $INSTANCE"
