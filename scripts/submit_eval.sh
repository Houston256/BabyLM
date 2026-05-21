#!/bin/bash
#PBS -N BabyLM_Eval
#PBS -l walltime=2:00:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_local=30gb:gpu_mem=20gb
#PBS -m abe
#PBS -j oe

# Standalone eval job: runs scripts/eval.sh in fast mode against the latest
# checkpoint under $DATADIR/checkpoints. Targets small GPUs (~20gb gpu_mem).
#
# Override the checkpoint by submitting with:
#   qsub -v CKPT_NAME=<dirname> scripts/submit_eval.sh

PROJECT_NAME="BabyLM"
HOMEDIR="/storage/praha1/home/$USER"
DATADIR="/storage/praha1/home/$USER/$PROJECT_NAME"

set -e
START_TIME=$(date +"%Y%m%d_%H%M%S")
export START_TIME

cleanup() {
    echo "=== Cleanup triggered at $(date) ==="
    if [ -d "$SCRATCHDIR/$PROJECT_NAME/eval/strict/results" ]; then
        echo "Syncing eval results back to $DATADIR/eval/strict/results/ ..."
        mkdir -p "$DATADIR/eval/strict/results"
        rsync -av "$SCRATCHDIR/$PROJECT_NAME/eval/strict/results/" "$DATADIR/eval/strict/results/"
    fi
    clean_scratch
}
trap cleanup EXIT

echo "Job started at $(date) on $(hostname)"
export TMPDIR=$SCRATCHDIR

echo "Copying project to scratch..."
rsync -av \
    --exclude "wandb/" \
    --exclude ".git/" \
    --exclude ".venv/" \
    --exclude "eval/" \
    --exclude "results_*/" \
    --exclude "checkpoints/" \
    --exclude "data/*.bin" \
    "$DATADIR/" "$SCRATCHDIR/$PROJECT_NAME/"

cd "$SCRATCHDIR/$PROJECT_NAME"

# eval/ stays in $DATADIR (large); symlink in so scripts/eval.sh resolves paths.
ln -snf "$DATADIR/eval" eval

# Resolve checkpoint: use $CKPT_NAME if provided, otherwise latest by mtime.
if [[ -n "${CKPT_NAME:-}" ]]; then
    CKPT_DIR="$DATADIR/checkpoints/$CKPT_NAME"
else
    CKPT_DIR=$(ls -1dt "$DATADIR/checkpoints"/*/ 2>/dev/null | head -n1)
    CKPT_DIR="${CKPT_DIR%/}"
fi

if [[ -z "$CKPT_DIR" || ! -d "$CKPT_DIR" ]]; then
    echo "no checkpoint found (CKPT_DIR=$CKPT_DIR)" >&2
    exit 1
fi
echo "Evaluating checkpoint: $CKPT_DIR"

echo "Setting up environment..."
source "$HOMEDIR/.profile" || true
export PATH="$HOMEDIR/.local/bin:$PATH"

uv venv --python 3.11
source .venv/bin/activate
uv sync --link-mode=copy

# Run both backends on fast eval set.
bash scripts/eval.sh "$CKPT_DIR" fast causal
bash scripts/eval.sh "$CKPT_DIR" fast mlm

echo "Eval completed successfully at $(date)"
