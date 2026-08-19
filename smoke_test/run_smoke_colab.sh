#!/usr/bin/env bash
set -euo pipefail

# One-command smoke test pipeline for Colab runtime.
# Usage:
#   bash /content/GemmaTest/smoke_test/run_smoke_colab.sh
# Optional env vars:
#   REPO_DIR=/content/GemmaTest
#   OPENI_REUSE=1
#   CSV_REUSE=1
#   HOLDOUT behavior: filter_openi.py now saves 20-case holdout to
#   /content/GemmaTest/smoke_test_holdout.csv and excludes it from smoke_subset.csv
#   RUN_EVAL=1
#   RUN_HOLDOUT_EVAL=1
#   HOLDOUT_CSV=/content/GemmaTest/smoke_test_holdout.csv
#   HOLDOUT_LIMIT=20
#   HOLDOUT_N=20
#   HOLDOUT_SEED=42
#   HF_TOKEN=<hf_token_with_model_access>
#   HF_LOGIN_REQUIRED=1
#   PROMPT_FOR_TOKENS=1
#   BATCH_SIZE=1
#   MAX_SEQ_LEN=512
#   SINGLE_VIEW=1
#   LOAD_IN_4BIT=1
#   TRAIN_EPOCHS=2
#   TRAIN_LR=2e-4
#   LORA_R=8
#   LORA_ALPHA=16
#   LORA_DROPOUT=0.05
#   LORA_TARGET_MODULES=q_proj,v_proj
#   ADAPTER_DIR=/content/drive/MyDrive/medgemma_smoke_output
#   EVAL_LIMIT=100
#   SAVE_RESULTS_VERSION=1
#   VERSION_LABEL=colab-t4
#   PUSH_RESULTS=1
#   GITHUB_REPO=MhdAna/GemmaTest
#   GITHUB_TOKEN=<token>
#   STATUS_FILE=/content/GemmaTest/smoke_status.txt

REPO_DIR="${REPO_DIR:-/content/GemmaTest}"
OPENI_DIR="${OPENI_DIR:-$REPO_DIR/openi}"
CSV_PATH="${CSV_PATH:-$REPO_DIR/smoke_subset.csv}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/smoke_output}"
OPENI_REUSE="${OPENI_REUSE:-1}"
CSV_REUSE="${CSV_REUSE:-1}"
RUN_EVAL="${RUN_EVAL:-0}"
RUN_HOLDOUT_EVAL="${RUN_HOLDOUT_EVAL:-0}"
HF_TOKEN="${HF_TOKEN:-}"
HF_LOGIN_REQUIRED="${HF_LOGIN_REQUIRED:-1}"
HF_PREFLIGHT_CHECK="${HF_PREFLIGHT_CHECK:-1}"
PROMPT_FOR_TOKENS="${PROMPT_FOR_TOKENS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
SINGLE_VIEW="${SINGLE_VIEW:-1}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"
TRAIN_LR="${TRAIN_LR:-2e-4}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,v_proj}"
ADAPTER_DIR="${ADAPTER_DIR:-$OUT_DIR}"
EVAL_LIMIT="${EVAL_LIMIT:-100}"
HOLDOUT_LIMIT="${HOLDOUT_LIMIT:-}"
HOLDOUT_CSV="${HOLDOUT_CSV:-$REPO_DIR/smoke_test_holdout.csv}"
HOLDOUT_N="${HOLDOUT_N:-20}"
HOLDOUT_SEED="${HOLDOUT_SEED:-42}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-/content/eval_output}"
HOLDOUT_OUTPUT_DIR="${HOLDOUT_OUTPUT_DIR:-$EVAL_OUTPUT_DIR/holdout}"
SAVE_RESULTS_VERSION="${SAVE_RESULTS_VERSION:-0}"
VERSION_LABEL="${VERSION_LABEL:-smoke}"
PUSH_RESULTS="${PUSH_RESULTS:-0}"
GITHUB_REPO="${GITHUB_REPO:-}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
STATUS_FILE="${STATUS_FILE:-$REPO_DIR/smoke_status.txt}"

TOTAL_STEPS=8
if [[ "$HF_PREFLIGHT_CHECK" == "1" ]]; then
  TOTAL_STEPS=$((TOTAL_STEPS + 1))
fi
if [[ "$RUN_EVAL" == "1" ]]; then
  TOTAL_STEPS=$((TOTAL_STEPS + 1))
fi
if [[ "$RUN_HOLDOUT_EVAL" == "1" ]]; then
  TOTAL_STEPS=$((TOTAL_STEPS + 1))
fi
if [[ "$SAVE_RESULTS_VERSION" == "1" ]]; then
  TOTAL_STEPS=$((TOTAL_STEPS + 1))
fi

CURRENT_STEP=0
update_status() {
  local message="$1"
  local percent_remaining="$2"
  mkdir -p "$(dirname "$STATUS_FILE")"
  cat > "$STATUS_FILE" <<EOF
step: ${CURRENT_STEP}/${TOTAL_STEPS}
percent_remaining: ${percent_remaining}
message: ${message}
updated_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
}

progress_step() {
  local label="$1"
  CURRENT_STEP=$((CURRENT_STEP + 1))
  local percent_complete=$((CURRENT_STEP * 100 / TOTAL_STEPS))
  local percent_remaining=$((100 - percent_complete))
  echo "[${CURRENT_STEP}/${TOTAL_STEPS} | ${percent_remaining}% remaining] ${label}"
  update_status "$label" "$percent_remaining"
}

update_status "Initializing" "100"

progress_step "Validate repository path"
if [[ ! -d "$REPO_DIR" ]]; then
  echo "ERROR: REPO_DIR not found: $REPO_DIR"
  echo "Clone repo first: git clone https://github.com/MhdAna/GemmaTest.git /content/GemmaTest"
  exit 1
fi

cd "$REPO_DIR"

progress_step "Install dependencies"
python -m pip install -U pip
python -m pip install -r requirements-macos.txt
python -m pip install -U "bitsandbytes>=0.46.1" evaluate rouge-score bert-score

progress_step "Prepare OpenI dataset"
mkdir -p "$OPENI_DIR"

download_openi_archive() {
  local target_file="$1"
  local primary_url="$2"
  local fallback_url="$3"
  local label="$4"

  local ok=0
  for url in "$primary_url" "$fallback_url"; do
    if [[ -z "$url" ]]; then
      continue
    fi

    echo "Downloading $label from: $url"
    if curl -L --fail -A "Mozilla/5.0" -o "$target_file" "$url"; then
      if file "$target_file" | grep -qi "gzip compressed data"; then
        ok=1
        break
      fi

      echo "WARN: Downloaded file is not gzip (likely HTML/error page)."
      echo "WARN: file output: $(file "$target_file")"
    else
      echo "WARN: Download failed from: $url"
    fi
  done

  if [[ "$ok" != "1" ]]; then
    echo "ERROR: Could not download a valid OpenI archive for $label."
    echo "ERROR: The OpenI endpoint is returning non-archive content."
    echo "Hint: set OPENI_REUSE=1 with an existing dataset folder, or place"
    echo "      the extracted OpenI files under: $OPENI_DIR"
    return 1
  fi
}

OPENI_READY=0
if [[ -d "$OPENI_DIR/ecgen-radiology" ]] && find "$OPENI_DIR" -type f -name "*.png" | head -n 1 >/dev/null; then
  OPENI_READY=1
fi

if [[ "$OPENI_REUSE" == "1" && "$OPENI_READY" == "1" ]]; then
  echo "Reusing existing OpenI data at $OPENI_DIR (download/extract skipped)."
else
  echo "Downloading OpenI archives"
  download_openi_archive \
    /content/NLMCXR_png.tgz \
    https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz \
    "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz?download=1" \
    "NLMCXR_png"

  download_openi_archive \
    /content/NLMCXR_reports.tgz \
    https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz \
    "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz?download=1" \
    "NLMCXR_reports"

  echo "Verify and extract OpenI archives"
  file /content/NLMCXR_png.tgz
  file /content/NLMCXR_reports.tgz
  tar -xzf /content/NLMCXR_png.tgz -C "$OPENI_DIR"
  tar -xzf /content/NLMCXR_reports.tgz -C "$OPENI_DIR"
fi

progress_step "Prepare smoke subset CSV"
DEFAULT_SUBSET_CSV="$REPO_DIR/smoke_subset.csv"
if [[ "$CSV_REUSE" == "1" && -f "$CSV_PATH" ]]; then
  echo "Reusing existing subset CSV at $CSV_PATH (build skipped)."
  if [[ "$CSV_PATH" != "$DEFAULT_SUBSET_CSV" ]]; then
    cp -f "$CSV_PATH" "$DEFAULT_SUBSET_CSV"
  fi
else
  python "$REPO_DIR/smoke_test/filter_openi.py" \
    --reports "$OPENI_DIR/ecgen-radiology" \
    --images "$OPENI_DIR" \
    --holdout-n "$HOLDOUT_N" \
    --seed "$HOLDOUT_SEED"

  # filter_openi.py writes to $REPO_DIR/smoke_subset.csv. Copy if caller wants a custom path.
  if [[ "$CSV_PATH" != "$DEFAULT_SUBSET_CSV" ]]; then
    mkdir -p "$(dirname "$CSV_PATH")"
    cp -f "$DEFAULT_SUBSET_CSV" "$CSV_PATH"
  fi
fi

if [[ ! -f "$CSV_PATH" ]]; then
  echo "ERROR: smoke subset CSV not found at $CSV_PATH"
  exit 1
fi

if [[ "$RUN_HOLDOUT_EVAL" == "1" && ! -f "$HOLDOUT_CSV" ]]; then
  echo "ERROR: holdout CSV not found at $HOLDOUT_CSV"
  echo "Rebuild CSV split with CSV_REUSE=0 or set HOLDOUT_CSV to an existing file."
  exit 1
fi

if [[ -z "$HF_TOKEN" && "$HF_LOGIN_REQUIRED" == "1" && "$PROMPT_FOR_TOKENS" == "1" ]]; then
  if [[ -t 0 ]]; then
    read -rsp "Enter HF_TOKEN (input hidden): " HF_TOKEN
    echo
  else
    echo "HF_TOKEN missing and no interactive terminal available for prompt."
    echo "Set HF_TOKEN env var or run with HF_LOGIN_REQUIRED=0."
  fi
fi

progress_step "Authenticate Hugging Face"
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
  progress_step "Preflight check: verify access to google/medgemma-4b-it"
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

progress_step "Run smoke training"
train_args=(
  --openi
  --images "$OPENI_DIR"
  --batch-size "$BATCH_SIZE"
  --max-seq-len "$MAX_SEQ_LEN"
  --epochs "$TRAIN_EPOCHS"
  --lr "$TRAIN_LR"
  --lora-r "$LORA_R"
  --lora-alpha "$LORA_ALPHA"
  --lora-dropout "$LORA_DROPOUT"
  --lora-target-modules "$LORA_TARGET_MODULES"
)

if [[ "$SINGLE_VIEW" == "1" ]]; then
  train_args+=(--single-view)
fi

if [[ "$LOAD_IN_4BIT" == "1" ]]; then
  train_args+=(--load-in-4bit)
fi

python "$REPO_DIR/smoke_test/smoke_train.py" "${train_args[@]}"

progress_step "Confirm output files"
ls -lah "$OUT_DIR"

if [[ "$RUN_EVAL" == "1" ]]; then
  progress_step "Run evaluation"
  python "$REPO_DIR/smoke_test/eval_colab.py" \
    --adapter-dir "$ADAPTER_DIR" \
    --csv "$CSV_PATH" \
    --images "$OPENI_DIR" \
    --output-dir "$EVAL_OUTPUT_DIR" \
    --use-4bit \
    --limit "$EVAL_LIMIT"

  echo "Evaluation outputs:"
  ls -lah "$EVAL_OUTPUT_DIR"
else
  echo "Evaluation skipped (set RUN_EVAL=1 to enable)"
fi

if [[ "$RUN_HOLDOUT_EVAL" == "1" ]]; then
  progress_step "Run holdout evaluation"
  holdout_eval_args=(
    --adapter-dir "$ADAPTER_DIR"
    --csv "$HOLDOUT_CSV"
    --images "$OPENI_DIR"
    --output-dir "$HOLDOUT_OUTPUT_DIR"
    --use-4bit
  )

  if [[ -n "$HOLDOUT_LIMIT" ]]; then
    holdout_eval_args+=(--limit "$HOLDOUT_LIMIT")
  fi

  python "$REPO_DIR/smoke_test/eval_colab.py" "${holdout_eval_args[@]}"

  echo "Holdout evaluation outputs:"
  ls -lah "$HOLDOUT_OUTPUT_DIR"
else
  echo "Holdout evaluation skipped (set RUN_HOLDOUT_EVAL=1 to enable)"
fi

echo "Done. Smoke test pipeline completed."

if [[ "$SAVE_RESULTS_VERSION" == "1" ]]; then
  if [[ "$PUSH_RESULTS" == "1" && -z "$GITHUB_TOKEN" && "$PROMPT_FOR_TOKENS" == "1" ]]; then
    if [[ -t 0 ]]; then
      read -rsp "Enter GITHUB_TOKEN (input hidden): " GITHUB_TOKEN
      echo
    else
      echo "GITHUB_TOKEN missing and no interactive terminal available for prompt."
      echo "Versioning will run, but push may fail without GITHUB_TOKEN."
    fi
  fi

  progress_step "Save versioned results snapshot to git"
  PUSH="$PUSH_RESULTS" \
  VERSION_LABEL="$VERSION_LABEL" \
  GITHUB_REPO="$GITHUB_REPO" \
  GITHUB_TOKEN="$GITHUB_TOKEN" \
  TRAIN_OPENI="1" \
  TRAIN_BATCH_SIZE="$BATCH_SIZE" \
  TRAIN_MAX_SEQ_LEN="$MAX_SEQ_LEN" \
  TRAIN_SINGLE_VIEW="$SINGLE_VIEW" \
  TRAIN_LOAD_IN_4BIT="$LOAD_IN_4BIT" \
  TRAIN_EPOCHS="$TRAIN_EPOCHS" \
  TRAIN_LR="$TRAIN_LR" \
  TRAIN_LORA_R="$LORA_R" \
  TRAIN_LORA_ALPHA="$LORA_ALPHA" \
  TRAIN_LORA_DROPOUT="$LORA_DROPOUT" \
  TRAIN_LORA_TARGET_MODULES="$LORA_TARGET_MODULES" \
  EVAL_ENABLED="$RUN_EVAL" \
  EVAL_LIMIT="$EVAL_LIMIT" \
  RESULTS_DIR="$OUT_DIR" \
  EVAL_DIR="$EVAL_OUTPUT_DIR" \
  CSV_PATH="$CSV_PATH" \
  bash "$REPO_DIR/smoke_test/version_results.sh"
fi

update_status "Completed" "0"
