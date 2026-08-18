#!/usr/bin/env bash
set -euo pipefail

# Prepare Colab runtime for MedGemma smoke test pipeline.
#
# Usage example:
#   REPO_URL="https://github.com/<user>/<repo>.git" \
#   HF_TOKEN="hf_xxx" \
#   GITHUB_TOKEN="ghp_xxx" \
#   bash smoke_test/prepare_colab.sh
#
# Optional env vars:
#   REPO_URL            (required) https clone URL of your repo
#   REPO_DIR            default: /content/GemmaTest
#   BRANCH              default: main
#   PROMPT_FOR_TOKENS   default: 1
#   SKIP_REPO_SYNC      default: 0 (set 1 if repo already cloned/synced)
#   AUTO_RUN            default: 0 (set 1 to run smoke pipeline immediately)
#   AUTO_PUSH           default: 1 (used when AUTO_RUN=1)
#   PUSH_TAG            default: 1 (used when AUTO_RUN=1 and AUTO_PUSH=1)
#   HF_TOKEN            optional at prep time, required by gated model run
#   GITHUB_TOKEN        optional, required only if pushing with token URL
#   GITHUB_REPO         optional owner/repo, used by version_results when PUSH=1
#   GIT_USER_NAME       optional git commit identity name
#   GIT_USER_EMAIL      optional git commit identity email

REPO_URL="${REPO_URL:-}"
REPO_DIR="${REPO_DIR:-/content/GemmaTest}"
BRANCH="${BRANCH:-main}"
PROMPT_FOR_TOKENS="${PROMPT_FOR_TOKENS:-1}"
SKIP_REPO_SYNC="${SKIP_REPO_SYNC:-0}"
AUTO_RUN="${AUTO_RUN:-0}"
AUTO_PUSH="${AUTO_PUSH:-1}"
PUSH_TAG="${PUSH_TAG:-1}"

HF_TOKEN="${HF_TOKEN:-}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
GITHUB_REPO="${GITHUB_REPO:-}"
GIT_USER_NAME="${GIT_USER_NAME:-}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-}"

OPENI_REUSE="${OPENI_REUSE:-1}"
CSV_REUSE="${CSV_REUSE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
SAVE_RESULTS_VERSION="${SAVE_RESULTS_VERSION:-1}"
PUSH_RESULTS="${PUSH_RESULTS:-0}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"
TRAIN_LR="${TRAIN_LR:-2e-4}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
EVAL_LIMIT="${EVAL_LIMIT:-100}"
VERSION_LABEL="${VERSION_LABEL:-colab-t4}"

if [[ -z "$REPO_URL" ]]; then
  echo "ERROR: REPO_URL is required."
  echo "Example: REPO_URL='https://github.com/<user>/<repo>.git' bash smoke_test/prepare_colab.sh"
  exit 1
fi

if [[ "$PROMPT_FOR_TOKENS" == "1" && -t 0 ]]; then
  if [[ -z "$HF_TOKEN" ]]; then
    read -rsp "Enter HF_TOKEN (leave empty to skip): " HF_TOKEN
    echo
  fi
  if [[ -z "$GITHUB_TOKEN" ]]; then
    read -rsp "Enter GITHUB_TOKEN (leave empty to skip): " GITHUB_TOKEN
    echo
  fi
fi

mkdir -p "$(dirname "$REPO_DIR")"

if [[ "$SKIP_REPO_SYNC" == "1" ]]; then
  if [[ ! -d "$REPO_DIR/.git" ]]; then
    echo "ERROR: SKIP_REPO_SYNC=1 but repo is missing at $REPO_DIR"
    exit 1
  fi
  echo "Skipping repo clone/sync (SKIP_REPO_SYNC=1)."
else
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
fi

cd "$REPO_DIR"
chmod +x smoke_test/run_smoke_colab.sh smoke_test/version_results.sh smoke_test/watch_smoke_status.sh smoke_test/bootstrap_colab.sh

if [[ -n "$GIT_USER_NAME" ]]; then
  git config user.name "$GIT_USER_NAME"
fi

if [[ -n "$GIT_USER_EMAIL" ]]; then
  git config user.email "$GIT_USER_EMAIL"
fi

ENV_FILE="$REPO_DIR/smoke_test/colab.env"
cat > "$ENV_FILE" <<EOF
export REPO_DIR="$REPO_DIR"
export REPO_URL="$REPO_URL"
export BRANCH="$BRANCH"

export HF_TOKEN="$HF_TOKEN"
export GITHUB_TOKEN="$GITHUB_TOKEN"
export GITHUB_REPO="$GITHUB_REPO"
export GIT_USER_NAME="$GIT_USER_NAME"
export GIT_USER_EMAIL="$GIT_USER_EMAIL"

export OPENI_REUSE="$OPENI_REUSE"
export CSV_REUSE="$CSV_REUSE"
export RUN_EVAL="$RUN_EVAL"
export SAVE_RESULTS_VERSION="$SAVE_RESULTS_VERSION"
export PUSH_RESULTS="$PUSH_RESULTS"

export LOAD_IN_4BIT="$LOAD_IN_4BIT"
export TRAIN_EPOCHS="$TRAIN_EPOCHS"
export TRAIN_LR="$TRAIN_LR"
export LORA_R="$LORA_R"
export LORA_ALPHA="$LORA_ALPHA"
export LORA_DROPOUT="$LORA_DROPOUT"

export EVAL_LIMIT="$EVAL_LIMIT"
export VERSION_LABEL="$VERSION_LABEL"
EOF

chmod 600 "$ENV_FILE"

echo
echo "Preparation complete."
echo "1) Load environment: source smoke_test/colab.env"
echo "2) Run pipeline: bash smoke_test/run_smoke_colab.sh"
echo "3) If PUSH_RESULTS=0, push manually after versioning."

if [[ "$AUTO_RUN" == "1" ]]; then
  echo
  echo "AUTO_RUN=1: starting smoke pipeline now..."
  bash "$REPO_DIR/smoke_test/run_smoke_colab.sh"

  if [[ "$AUTO_PUSH" == "1" ]]; then
    if [[ -z "$GITHUB_TOKEN" ]]; then
      echo "ERROR: AUTO_PUSH=1 but GITHUB_TOKEN is empty."
      echo "Set GITHUB_TOKEN or run with AUTO_PUSH=0."
      exit 1
    fi

    branch="$BRANCH"
    repo_no_proto="${REPO_URL#https://}"
    auth_url="https://${GITHUB_TOKEN}@${repo_no_proto}"

    echo "Pushing branch: $branch"
    git push "$auth_url" "$branch"

    if [[ "$PUSH_TAG" == "1" ]]; then
      latest_tag="$(git tag --sort=-creatordate | head -n 1 || true)"
      if [[ -n "$latest_tag" ]]; then
        echo "Pushing latest tag: $latest_tag"
        git push "$auth_url" "$latest_tag"
      else
        echo "No tags found to push."
      fi
    fi

    echo "Auto run + push complete."
  else
    echo "AUTO_PUSH=0, skipping git push."
  fi
fi
