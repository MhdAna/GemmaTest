#!/usr/bin/env bash
set -euo pipefail

# Bootstrap Colab from a clean runtime with no local repo folder.
# This script clones/pulls the repo, then runs prepare_colab.sh.
#
# Usage (from Colab):
#   REPO_URL="https://github.com/MhdAna/GemmaTest.git" \
#   BRANCH="main" \
#   bash <(curl -fsSL https://raw.githubusercontent.com/MhdAna/GemmaTest/main/smoke_test/bootstrap_colab.sh)

REPO_URL="${REPO_URL:-https://github.com/MhdAna/GemmaTest.git}"
REPO_DIR="${REPO_DIR:-/content/GemmaTest}"
BRANCH="${BRANCH:-main}"
PROMPT_FOR_TOKENS="${PROMPT_FOR_TOKENS:-1}"

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
  PROMPT_FOR_TOKENS="$PROMPT_FOR_TOKENS" \
  SKIP_REPO_SYNC="1" \
  AUTO_RUN="${AUTO_RUN:-0}" \
  AUTO_PUSH="${AUTO_PUSH:-1}" \
  PUSH_TAG="${PUSH_TAG:-1}" \
  HF_TOKEN="${HF_TOKEN:-}" \
  GITHUB_TOKEN="${GITHUB_TOKEN:-}" \
  GITHUB_REPO="${GITHUB_REPO:-}" \
  GIT_USER_NAME="${GIT_USER_NAME:-}" \
  GIT_USER_EMAIL="${GIT_USER_EMAIL:-}" \
  OPENI_REUSE="${OPENI_REUSE:-1}" \
  CSV_REUSE="${CSV_REUSE:-1}" \
  RUN_EVAL="${RUN_EVAL:-1}" \
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
