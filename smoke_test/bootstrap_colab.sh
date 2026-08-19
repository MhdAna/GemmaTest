#!/usr/bin/env bash
set -euo pipefail

# Bootstrap Colab from a clean runtime with no local repo folder.
# This script clones/pulls the repo, then runs prepare_colab.sh.
#
# Usage (from Colab):
#   REPO_URL="https://github.com/MhdAna/GemmaTest.git" \
#   BRANCH="main" \
#   bash <(curl -fsSL https://raw.githubusercontent.com/MhdAna/GemmaTest/main/smoke_test/bootstrap_colab.sh)
#
# Notes:
#   - Provide HF_TOKEN via environment when running this script.
#   - GITHUB_TOKEN is optional and only needed when push is enabled later.

REPO_URL="${REPO_URL:-https://github.com/MhdAna/GemmaTest.git}"
REPO_DIR="${REPO_DIR:-/content/GemmaTest}"
BRANCH="${BRANCH:-main}"
HF_TOKEN="${HF_TOKEN:-}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

if [[ -z "$HF_TOKEN" ]]; then
  echo "ERROR: HF_TOKEN is required."
  echo "Set HF_TOKEN in environment before running bootstrap."
  exit 1
fi

mkdir -p "$(dirname "$REPO_DIR")"

if [[ -d "$REPO_DIR/.git" ]]; then
  echo "Repo exists, syncing latest changes..."
  git -C "$REPO_DIR" remote set-url origin "$REPO_URL"
  git -C "$REPO_DIR" fetch origin
  git -C "$REPO_DIR" checkout "$BRANCH"
  git -C "$REPO_DIR" pull --rebase origin "$BRANCH"
else
  echo "Cloning repository..."
  git clone "$REPO_URL" "$REPO_DIR"
  git -C "$REPO_DIR" checkout "$BRANCH"
fi

cd "$REPO_DIR"
chmod +x smoke_test/prepare_colab.sh smoke_test/watch_smoke_status.sh || true

exec env \
  REPO_URL="$REPO_URL" \
  REPO_DIR="$REPO_DIR" \
  BRANCH="$BRANCH" \
  PROMPT_FOR_TOKENS="0" \
  SKIP_REPO_SYNC="1" \
  AUTO_RUN="${AUTO_RUN:-0}" \
  AUTO_PUSH="${AUTO_PUSH:-1}" \
  PUSH_TAG="${PUSH_TAG:-1}" \
  HF_TOKEN="$HF_TOKEN" \
  GITHUB_TOKEN="$GITHUB_TOKEN" \
  GITHUB_REPO="${GITHUB_REPO:-}" \
  GIT_USER_NAME="${GIT_USER_NAME:-}" \
  GIT_USER_EMAIL="${GIT_USER_EMAIL:-}" \
  OPENI_REUSE="${OPENI_REUSE:-1}" \
  CSV_REUSE="${CSV_REUSE:-1}" \
  RUN_EVAL="${RUN_EVAL:-1}" \
  RUN_HOLDOUT_EVAL="${RUN_HOLDOUT_EVAL:-0}" \
  HOLDOUT_LIMIT="${HOLDOUT_LIMIT:-}" \
  HOLDOUT_N="${HOLDOUT_N:-20}" \
  HOLDOUT_SEED="${HOLDOUT_SEED:-42}" \
  SAVE_RESULTS_VERSION="${SAVE_RESULTS_VERSION:-1}" \
  PUSH_RESULTS="${PUSH_RESULTS:-0}" \
  LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}" \
  TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}" \
  TRAIN_LR="${TRAIN_LR:-2e-4}" \
  LORA_R="${LORA_R:-8}" \
  LORA_ALPHA="${LORA_ALPHA:-16}" \
  LORA_DROPOUT="${LORA_DROPOUT:-0.05}" \
  EVAL_LIMIT="${EVAL_LIMIT:-100}" \
  VERSION_LABEL="${VERSION_LABEL:-colab-t4}" \
  bash "$REPO_DIR/smoke_test/prepare_colab.sh"
