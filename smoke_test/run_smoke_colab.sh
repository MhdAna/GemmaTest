#!/usr/bin/env bash
set -euo pipefail

# One-command smoke test pipeline for Colab runtime.
# Usage:
#   bash /content/GemmaTest/smoke_test/run_smoke_colab.sh
# Optional env vars:
#   REPO_DIR=/content/GemmaTest
#   OPENI_REUSE=1
#   CSV_REUSE=1
#   RUN_EVAL=1
#   HF_TOKEN=<hf_token_with_model_access>
#   HF_LOGIN_REQUIRED=1
#   ADAPTER_DIR=/content/drive/MyDrive/medgemma_smoke_output
#   EVAL_LIMIT=100
#   SAVE_RESULTS_VERSION=1
#   VERSION_LABEL=colab-t4
#   PUSH_RESULTS=1
#   GITHUB_REPO=MhdAna/GemmaTest
#   GITHUB_TOKEN=<token>

REPO_DIR="${REPO_DIR:-/content/GemmaTest}"
OPENI_DIR="${OPENI_DIR:-$REPO_DIR/openi}"
CSV_PATH="${CSV_PATH:-$REPO_DIR/smoke_subset.csv}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/smoke_output}"
OPENI_REUSE="${OPENI_REUSE:-1}"
CSV_REUSE="${CSV_REUSE:-1}"
RUN_EVAL="${RUN_EVAL:-0}"
HF_TOKEN="${HF_TOKEN:-}"
HF_LOGIN_REQUIRED="${HF_LOGIN_REQUIRED:-1}"
HF_PREFLIGHT_CHECK="${HF_PREFLIGHT_CHECK:-1}"
ADAPTER_DIR="${ADAPTER_DIR:-/content/drive/MyDrive/medgemma_smoke_output}"
EVAL_LIMIT="${EVAL_LIMIT:-100}"
SAVE_RESULTS_VERSION="${SAVE_RESULTS_VERSION:-0}"
VERSION_LABEL="${VERSION_LABEL:-smoke}"
PUSH_RESULTS="${PUSH_RESULTS:-0}"
GITHUB_REPO="${GITHUB_REPO:-}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

echo "[1/8] Validate repository path"
if [[ ! -d "$REPO_DIR" ]]; then
  echo "ERROR: REPO_DIR not found: $REPO_DIR"
  echo "Clone repo first: git clone https://github.com/MhdAna/GemmaTest.git /content/GemmaTest"
  exit 1
fi

cd "$REPO_DIR"

echo "[2/8] Install dependencies"
python -m pip install -U pip
python -m pip install -r requirements-macos.txt
python -m pip install -U "bitsandbytes>=0.46.1" evaluate rouge-score bert-score

echo "[3/8] Prepare OpenI dataset"
mkdir -p "$OPENI_DIR"

OPENI_READY=0
if [[ -d "$OPENI_DIR/ecgen-radiology" ]] && find "$OPENI_DIR" -type f -name "*.png" | head -n 1 >/dev/null; then
  OPENI_READY=1
fi

if [[ "$OPENI_REUSE" == "1" && "$OPENI_READY" == "1" ]]; then
  echo "Reusing existing OpenI data at $OPENI_DIR (download/extract skipped)."
else
  echo "Downloading OpenI archives"
  curl -L --fail -o /content/NLMCXR_png.tgz https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz
  curl -L --fail -o /content/NLMCXR_reports.tgz https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz

  echo "[4/8] Verify and extract OpenI archives"
  file /content/NLMCXR_png.tgz
  file /content/NLMCXR_reports.tgz
  tar -xzf /content/NLMCXR_png.tgz -C "$OPENI_DIR"
  tar -xzf /content/NLMCXR_reports.tgz -C "$OPENI_DIR"
fi

echo "[5/8] Prepare smoke subset CSV"
if [[ "$CSV_REUSE" == "1" && -f "$CSV_PATH" ]]; then
  echo "Reusing existing subset CSV at $CSV_PATH (build skipped)."
else
  python "$REPO_DIR/smoke_test/filter_openi.py" \
    --reports "$OPENI_DIR/ecgen-radiology" \
    --images "$OPENI_DIR"

  # filter_openi.py writes to $REPO_DIR/smoke_subset.csv. Copy if caller wants a custom path.
  DEFAULT_SUBSET_CSV="$REPO_DIR/smoke_subset.csv"
  if [[ "$CSV_PATH" != "$DEFAULT_SUBSET_CSV" ]]; then
    mkdir -p "$(dirname "$CSV_PATH")"
    cp -f "$DEFAULT_SUBSET_CSV" "$CSV_PATH"
  fi
fi

if [[ ! -f "$CSV_PATH" ]]; then
  echo "ERROR: smoke subset CSV not found at $CSV_PATH"
  exit 1
fi

echo "[6/8] Authenticate Hugging Face (for gated model access)"
if [[ -n "$HF_TOKEN" ]]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  export HF_TOKEN="$HF_TOKEN"
  python - <<'PY'
import os
from huggingface_hub import login

token = os.environ.get("HF_TOKEN", "")
if token:
    login(token=token, add_to_git_credential=False)
PY
elif [[ "$HF_LOGIN_REQUIRED" == "1" ]]; then
  echo "ERROR: HF_TOKEN is required for gated model google/medgemma-4b-it."
  echo "Get access at: https://huggingface.co/google/medgemma-4b-it"
  echo "Then run with: HF_TOKEN=hf_xxx bash $REPO_DIR/smoke_test/run_smoke_colab.sh"
  exit 1
else
  echo "HF_TOKEN not set; continuing without login (may fail if model access is gated)."
fi

if [[ "$HF_PREFLIGHT_CHECK" == "1" ]]; then
  echo "[6/8] Preflight check: verify access to google/medgemma-4b-it"
  python - <<'PY'
import os
import sys
from huggingface_hub import HfApi

model_id = "google/medgemma-4b-it"
token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

try:
    HfApi().model_info(model_id, token=token)
    print(f"HF access OK: {model_id}")
except Exception as e:
    print(f"ERROR: Hugging Face access check failed for {model_id}.")
    print("Make sure you requested access at: https://huggingface.co/google/medgemma-4b-it")
    print("Then provide HF_TOKEN when running this script.")
    print(f"Details: {e}")
    sys.exit(1)
PY
fi

echo "[7/8] Run smoke training"
python "$REPO_DIR/smoke_test/smoke_train.py" \
  --openi \
  --images "$OPENI_DIR" \
  --single-view \
  --batch-size 1 \
  --max-seq-len 512 \
  --load-in-4bit

echo "[8/8] Confirm output files"
ls -lah "$OUT_DIR"

if [[ "$RUN_EVAL" == "1" ]]; then
  echo "[8/8] Run evaluation"
  python "$REPO_DIR/smoke_test/eval_colab.py" \
    --adapter-dir "$ADAPTER_DIR" \
    --csv "$CSV_PATH" \
    --images "$OPENI_DIR" \
    --output-dir /content/eval_output \
    --use-4bit \
    --limit "$EVAL_LIMIT"

  echo "Evaluation outputs:"
  ls -lah /content/eval_output
else
  echo "[8/8] Evaluation skipped (set RUN_EVAL=1 to enable)"
fi

echo "Done. Smoke test pipeline completed."

if [[ "$SAVE_RESULTS_VERSION" == "1" ]]; then
  echo "Saving versioned results snapshot to git"
  PUSH="$PUSH_RESULTS" \
  VERSION_LABEL="$VERSION_LABEL" \
  GITHUB_REPO="$GITHUB_REPO" \
  GITHUB_TOKEN="$GITHUB_TOKEN" \
  RESULTS_DIR="$OUT_DIR" \
  EVAL_DIR="/content/eval_output" \
  CSV_PATH="$CSV_PATH" \
  bash "$REPO_DIR/smoke_test/version_results.sh"
fi
