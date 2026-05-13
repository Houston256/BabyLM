#!/usr/bin/env bash
# One-time eval data setup:
#   1. unzip the password-protected EWoK fast set
#   2. download nltk's punkt_tab (used by EWoK filter script)
#   3. download + filter the full EWoK dataset
#
# Prereqs: HF_TOKEN in .env (or env), and you've accepted EWoK terms at
#   https://huggingface.co/datasets/ewok-core/ewok-core-1.0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a; . .env; set +a
fi

EWOK_FAST_DIR="eval/strict/evaluation_data/fast_eval/ewok_fast"
if [[ ! -d "$EWOK_FAST_DIR" ]]; then
  echo "[setup] unzipping EWoK fast set"
  (cd eval/strict && unzip -q -P BabyLM2025 evaluation_data/fast_eval/ewok_fast.zip)
fi

EWOK_FULL_DIR="eval/strict/evaluation_data/full_eval/ewok_filtered"
if [[ ! -d "$EWOK_FULL_DIR" ]]; then
  echo "[setup] fetching nltk punkt_tab"
  # On macOS, uv-managed Python lacks system CA certs (no Install Certificates.command),
  # so nltk's HTTPS download fails with CERTIFICATE_VERIFY_FAILED. Point it at certifi's bundle.
  if [[ "$(uname)" == "Darwin" ]]; then
    export SSL_CERT_FILE="$(uv run python -c 'import certifi; print(certifi.where())')"
  fi
  uv run python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

  echo "[setup] downloading + filtering full EWoK (requires HF_TOKEN + accepted terms)"
  (cd eval/strict && uv run --project "$ROOT" python -m evaluation_pipeline.ewok.dl_and_filter)
fi

echo "[setup] done"
