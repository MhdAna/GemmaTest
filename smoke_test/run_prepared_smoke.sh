#!/usr/bin/env bash
set -euo pipefail

# Run the smoke pipeline from a prepared Colab environment and optionally push.
#
# Usage:
#   bash smoke_test/run_prepared_smoke.sh
#
# Optional env vars:
#   ENV_FILE=smoke_test/colab.env
#   AUTO_PUSH=1
#   PUSH_TAG=1

REPO_DIR="${REPO_DIR:-/content/GemmaTest}"
ENV_FILE="${ENV_FILE:-$REPO_DIR/smoke_test/colab.env}"
AUTO_PUSH="${AUTO_PUSH:-1}"
PUSH_TAG="${PUSH_TAG:-1}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: env file not found: $ENV_FILE"
  echo "Run: bash smoke_test/prepare_colab.sh"
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "ERROR: git repo not found: $REPO_DIR"
  exit 1
fi

cd "$REPO_DIR"

echo "Running smoke pipeline..."
bash "$REPO_DIR/smoke_test/run_smoke_colab.sh"

if [[ "$AUTO_PUSH" != "1" ]]; then
  echo "AUTO_PUSH=0, skipping git push."
  exit 0
fi

if [[ -z "${REPO_URL:-}" ]]; then
  echo "ERROR: REPO_URL missing in environment."
  echo "Re-run prepare script with REPO_URL set."
  exit 1
fi

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "ERROR: GITHUB_TOKEN missing for authenticated push."
  echo "Set GITHUB_TOKEN in prepare step or export it before running."
  exit 1
fi

branch="${BRANCH:-main}"
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

echo "Run + push complete."
