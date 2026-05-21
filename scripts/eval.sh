#!/usr/bin/env bash
# usage: scripts/eval.sh <checkpoint_dir> [fast|full] [causal|mlm]
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <checkpoint_dir> [fast|full] [causal|mlm]" >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
EVAL_DIR="$REPO_ROOT/eval/strict"

CKPT_DIR=$(cd "$1" && pwd)
MODE=${2:-fast}
BACKEND=${3:-causal}

echo "checkpoint: $CKPT_DIR"
echo "mode:       $MODE"
echo "backend:    $BACKEND"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

cd "$EVAL_DIR"
case "$MODE" in
    fast) bash scripts/eval_zero_shot_fast.sh "$CKPT_DIR" main "$BACKEND" ;;
    full) bash scripts/eval_zero_shot.sh      "$CKPT_DIR"      "$BACKEND" ;;
    *)    echo "unknown mode: $MODE (expected fast|full)" >&2; exit 2 ;;
esac
