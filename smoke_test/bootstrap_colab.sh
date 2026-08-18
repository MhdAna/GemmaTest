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
#   - If PROMPT_FOR_TOKENS=1, this script always prompts for HF_TOKEN and
#     optionally prompts for GITHUB_TOKEN.
#   - In non-interactive shells (e.g., some %%bash executions), prompts are not possible.
#     In that case, set HF_TOKEN in a Python cell first and rerun with PROMPT_FOR_TOKENS=0.

REPO_URL="${REPO_URL:-https://github.com/MhdAna/GemmaTest.git}"
REPO_DIR="${REPO_DIR:-/content/GemmaTest}"
BRANCH="${BRANCH:-main}"
PROMPT_FOR_TOKENS="${PROMPT_FOR_TOKENS:-1}"
HF_TOKEN="${HF_TOKEN:-}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

if [[ "$PROMPT_FOR_TOKENS" == "1" ]]; then
  if [[ -t 0 ]]; then
    # Always ask for HF token in interactive mode to avoid passing it as an argument.
    HF_TOKEN=""
    while [[ -z "$HF_TOKEN" ]]; do
      read -rsp "Enter HF_TOKEN (input hidden, required): " HF_TOKEN
      echo
      if [[ -z "$HF_TOKEN" ]]; then
        echo "HF_TOKEN cannot be empty for MedGemma gated access."
      fi
    done
    if [[ -z "$GITHUB_TOKEN" ]]; then
      read -rsp "Enter GITHUB_TOKEN (optional, input hidden): " GITHUB_TOKEN
      echo
    fi
  elif [[ -z "$HF_TOKEN" ]]; then
    echo "HF_TOKEN missing and no interactive terminal is available for prompt."
    echo "Set HF_TOKEN before running bootstrap. Example in a Python cell:"
    echo "  import os"
    echo "  from getpass import getpass"
    echo "  os.environ['HF_TOKEN'] = getpass('HF token: ')"
    echo "Then run bootstrap with PROMPT_FOR_TOKENS=0."
    exit 1
  fi
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
