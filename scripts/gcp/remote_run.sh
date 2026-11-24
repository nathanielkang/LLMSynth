#!/usr/bin/env bash
# Run full RelDDPM + TabDiff grid on a GCP VM with persistent logging.
set -euo pipefail

PROJECT="${GCP_PROJECT:-knu-fog-experiments}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE="${GCP_INSTANCE:-llmsynth-gpu}"
REMOTE_DIR="${REMOTE_DIR:-~/llmsynth/2_Code}"
OUTPUT_DIR="${OUTPUT_DIR:-results_gcp_relddpm_tabdiff_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-200}"
SEEDS="${SEEDS:-3}"

CMD=$(cat <<EOF
set -e
cd ${REMOTE_DIR}
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -r third_party/RelDDPM/requirements.txt || true
python scripts/smoke_relddpm_tabdiff.py
python run_experiments.py \\
  --method RelDDPM TabDiff \\
  --dataset all \\
  --seeds ${SEEDS} \\
  --epochs ${EPOCHS} \\
  --use-mock \\
  --output-dir ${OUTPUT_DIR} \\
  2>&1 | tee ${OUTPUT_DIR}.log
echo "DONE -> ${OUTPUT_DIR}/results_raw.csv"
EOF
)

gcloud compute ssh --project "$PROJECT" --zone "$ZONE" "$INSTANCE" --command "$CMD"
