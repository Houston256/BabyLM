#!/bin/bash
#PBS -N BabyLM_Full_Pipeline
#PBS -l walltime=1:00:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:scratch_local=50gb:gpu_mem=40gb
#PBS -m abe
#PBS -j oe

# --- Configuration (Automatic & Universal) ---
PROJECT_NAME="BabyLM"
# The cluster script uses the system's $USER variable automatically
HOMEDIR="/storage/praha1/home/$USER"
DATADIR="/storage/praha1/home/$USER/$PROJECT_NAME"

# --- Setup ---
set -e
START_TIME=$(date +"%Y%m%d_%H%M%S")
export START_TIME

cleanup() {
    echo "=== Cleanup triggered at $(date) ==="
    if [ -d "$SCRATCHDIR/$PROJECT_NAME/checkpoints" ]; then
        echo "Syncing checkpoints back to $DATADIR/checkpoints/ ..."
        mkdir -p "$DATADIR/checkpoints"
        rsync -av "$SCRATCHDIR/$PROJECT_NAME/checkpoints/" "$DATADIR/checkpoints/"
    fi
    clean_scratch
}
trap cleanup EXIT

echo "Job started at $(date) on $(hostname)"
export TMPDIR=$SCRATCHDIR

# 1. Copy project to scratch
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

# eval/ is excluded from rsync (large eval data lives in $DATADIR). Symlink it
# in so train.py's inline eval block can resolve scripts/eval.sh -> eval/strict/.
ln -snf "$DATADIR/eval" eval

# 2. Setup Environment
echo "Setting up environment..."
source "$HOMEDIR/.profile" || true
export PATH="$HOMEDIR/.local/bin:$PATH"

uv venv --python 3.11
source .venv/bin/activate
uv sync --link-mode=copy

# --- Execution ---

# 3. Tokenize
# DATA_ARGS controls corpus preprocessing. Three useful modes:
#   raw (current default): -v DATA_ARGS=""           (no preprocessing, HF flat dataset)
#   strip+utterance SEP:   -v DATA_ARGS="--strip-speaker-tags --insert-sep"
#   conversation SEP:      -v DATA_ARGS="--source-mode raw --raw-dir data/raw"
# The bin is rebuilt in-scratch every job, so DATA_ARGS is what trains.
DATA_ARGS=${DATA_ARGS:-}

# Always pull the raw per-source txt files (55 MB total) so --source-mode raw works.
mkdir -p data/raw
for src in childes simple_wiki gutenberg open_subtitles bnc_spoken switchboard; do
    if [ ! -f "data/raw/${src}.train.txt" ]; then
        curl -sL "https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict-Small/resolve/main/${src}.train.txt" \
            -o "data/raw/${src}.train.txt"
    fi
done

uv run python main.py train-tokenizer --vocab-size 8192
# shellcheck disable=SC2086  # intentional word splitting on DATA_ARGS
uv run python main.py tokenize-corpus --tokenizer models/tokenizer.json --output data/train.bin $DATA_ARGS

# 4. Pretrain + inline eval (logs eval metrics to the live wandb run before finishing).
# All arch/training hyperparams default to the values in add_pretrain_args().
# Override any subset via ARGS:
#   qsub -v ARGS="--run-name foo --pos-emb rope --rope-base 1000" scripts/submit_babylm.sh
# To ablate preprocessing (e.g. the old raw bin), also pass:
#   -v DATA_ARGS=""
ARGS=${ARGS:-}

# shellcheck disable=SC2086  # intentional word splitting on ARGS
uv run python main.py pretrain \
    --output-dir checkpoints/ \
    --wandb \
    --eval fast \
    $ARGS

echo "All steps completed successfully at $(date)"
