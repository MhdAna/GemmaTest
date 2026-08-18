#!/usr/bin/env bash
set -euo pipefail

# Save smoke-test artifacts as a versioned snapshot in git.
# Usage:
#   bash /content/GemmaTest/smoke_test/version_results.sh
# Optional env vars:
#   REPO_DIR=/content/GemmaTest
#   RESULTS_DIR=/content/GemmaTest/smoke_output
#   EVAL_DIR=/content/eval_output
#   CSV_PATH=/content/GemmaTest/smoke_subset.csv
#   VERSION_LABEL=colab-t4
#   BRANCH=main
#   CREATE_TAG=1
#   PUSH=0
#   GITHUB_REPO=MhdAna/GemmaTest
#   GITHUB_TOKEN=<token>
#   TRAIN_OPENI=1
#   TRAIN_BATCH_SIZE=1
#   TRAIN_MAX_SEQ_LEN=512
#   TRAIN_SINGLE_VIEW=1
#   TRAIN_LOAD_IN_4BIT=1
#   TRAIN_EPOCHS=2
#   TRAIN_LR=2e-4
#   TRAIN_LORA_R=8
#   TRAIN_LORA_ALPHA=16
#   TRAIN_LORA_DROPOUT=0.05
#   TRAIN_LORA_TARGET_MODULES=q_proj,v_proj
#   EVAL_ENABLED=0
#   EVAL_LIMIT=100
#   GIT_USER_NAME="Your Name"
#   GIT_USER_EMAIL="you@example.com"

REPO_DIR="${REPO_DIR:-/content/GemmaTest}"
RESULTS_DIR="${RESULTS_DIR:-/content/GemmaTest/smoke_output}"
EVAL_DIR="${EVAL_DIR:-/content/eval_output}"
CSV_PATH="${CSV_PATH:-/content/GemmaTest/smoke_subset.csv}"
VERSION_LABEL="${VERSION_LABEL:-smoke}"
BRANCH="${BRANCH:-main}"
CREATE_TAG="${CREATE_TAG:-1}"
PUSH="${PUSH:-0}"
GITHUB_REPO="${GITHUB_REPO:-}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
TRAIN_OPENI="${TRAIN_OPENI:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
TRAIN_MAX_SEQ_LEN="${TRAIN_MAX_SEQ_LEN:-512}"
TRAIN_SINGLE_VIEW="${TRAIN_SINGLE_VIEW:-1}"
TRAIN_LOAD_IN_4BIT="${TRAIN_LOAD_IN_4BIT:-1}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"
TRAIN_LR="${TRAIN_LR:-2e-4}"
TRAIN_LORA_R="${TRAIN_LORA_R:-8}"
TRAIN_LORA_ALPHA="${TRAIN_LORA_ALPHA:-16}"
TRAIN_LORA_DROPOUT="${TRAIN_LORA_DROPOUT:-0.05}"
TRAIN_LORA_TARGET_MODULES="${TRAIN_LORA_TARGET_MODULES:-q_proj,v_proj}"
EVAL_ENABLED="${EVAL_ENABLED:-0}"
EVAL_LIMIT="${EVAL_LIMIT:-100}"
GIT_USER_NAME="${GIT_USER_NAME:-}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-}"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "ERROR: Not a git repo: $REPO_DIR"
  exit 1
fi

if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "ERROR: RESULTS_DIR not found: $RESULTS_DIR"
  exit 1
fi

sanitize() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+|-+$//g'
}

cd "$REPO_DIR"

if [[ -n "$GIT_USER_NAME" ]]; then
  git config user.name "$GIT_USER_NAME"
fi

if [[ -n "$GIT_USER_EMAIL" ]]; then
  git config user.email "$GIT_USER_EMAIL"
fi

if [[ -z "$(git config user.name || true)" || -z "$(git config user.email || true)" ]]; then
  echo "ERROR: git user identity is not configured for commits."
  echo "Set repo-local identity with:"
  echo "  GIT_USER_NAME='Your Name' GIT_USER_EMAIL='you@example.com' bash smoke_test/version_results.sh"
  echo "Or configure git manually with:"
  echo "  git config user.name 'Your Name'"
  echo "  git config user.email 'you@example.com'"
  exit 1
fi

stamp="$(date -u +%Y%m%d-%H%M%S)"
label="$(sanitize "$VERSION_LABEL")"
version_id="${stamp}-${label}"
version_dir="results/versions/${version_id}"

mkdir -p "$version_dir"
cp -a "$RESULTS_DIR" "$version_dir/smoke_output"

if [[ -d "$EVAL_DIR" ]]; then
  cp -a "$EVAL_DIR" "$version_dir/eval_output"
else
  echo "INFO: EVAL_DIR not found, skipping: $EVAL_DIR"
fi

if [[ -f "$CSV_PATH" ]]; then
  cp -a "$CSV_PATH" "$version_dir/smoke_subset.csv"
else
  echo "INFO: CSV_PATH not found, skipping: $CSV_PATH"
fi

base_sha="$(git rev-parse HEAD)"
cat > "$version_dir/MANIFEST.txt" <<EOF
version_id: $version_id
created_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)
base_commit: $base_sha
results_dir: $RESULTS_DIR
eval_dir: $EVAL_DIR
csv_path: $CSV_PATH
train_openi: $TRAIN_OPENI
train_batch_size: $TRAIN_BATCH_SIZE
train_max_seq_len: $TRAIN_MAX_SEQ_LEN
train_single_view: $TRAIN_SINGLE_VIEW
train_load_in_4bit: $TRAIN_LOAD_IN_4BIT
train_epochs: $TRAIN_EPOCHS
train_lr: $TRAIN_LR
train_lora_r: $TRAIN_LORA_R
train_lora_alpha: $TRAIN_LORA_ALPHA
train_lora_dropout: $TRAIN_LORA_DROPOUT
train_lora_target_modules: $TRAIN_LORA_TARGET_MODULES
eval_enabled: $EVAL_ENABLED
eval_limit: $EVAL_LIMIT
EOF

git add "$version_dir"

if git diff --cached --quiet; then
  echo "No new result files to commit."
  exit 0
fi

commit_msg="results: add smoke snapshot ${version_id}"
git commit -m "$commit_msg"

echo "Created commit: $(git rev-parse --short HEAD)"

tag_name="results-${version_id}"
if [[ "$CREATE_TAG" == "1" ]]; then
  git tag -a "$tag_name" -m "Smoke results $version_id"
  echo "Created tag: $tag_name"
fi

if [[ "$PUSH" == "1" ]]; then
  if [[ -n "$GITHUB_TOKEN" && -n "$GITHUB_REPO" ]]; then
    push_url="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git"
    git push "$push_url" HEAD:"$BRANCH"
    if [[ "$CREATE_TAG" == "1" ]]; then
      git push "$push_url" "$tag_name"
    fi
  else
    git push origin HEAD:"$BRANCH"
    if [[ "$CREATE_TAG" == "1" ]]; then
      git push origin "$tag_name"
    fi
  fi
  echo "Pushed commit and tags."
else
  echo "PUSH=0, local commit/tag only."
fi

echo "Version snapshot stored at: $version_dir"
