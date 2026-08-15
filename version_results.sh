#!/usr/bin/env bash
set -euo pipefail

# Save smoke-test artifacts as a versioned snapshot in git.
# Usage:
#   bash /content/GemmaTest/smoke_test/version_results.sh
# Optional env vars:
#   REPO_DIR=/content/GemmaTest
#   RESULTS_DIR=/content/smoke_output
#   EVAL_DIR=/content/eval_output
#   CSV_PATH=/content/smoke_subset.csv
#   VERSION_LABEL=colab-t4
#   BRANCH=main
#   CREATE_TAG=1
#   PUSH=0
#   GITHUB_REPO=MhdAna/GemmaTest
#   GITHUB_TOKEN=<token>

REPO_DIR="${REPO_DIR:-/content/GemmaTest}"
RESULTS_DIR="${RESULTS_DIR:-/content/smoke_output}"
EVAL_DIR="${EVAL_DIR:-/content/eval_output}"
CSV_PATH="${CSV_PATH:-/content/smoke_subset.csv}"
VERSION_LABEL="${VERSION_LABEL:-smoke}"
BRANCH="${BRANCH:-main}"
CREATE_TAG="${CREATE_TAG:-1}"
PUSH="${PUSH:-0}"
GITHUB_REPO="${GITHUB_REPO:-}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

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
