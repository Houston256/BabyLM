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
uv run python main.py train-tokenizer --vocab-size 8192
uv run python main.py tokenize-corpus --tokenizer models/tokenizer.json --output data/train.bin

# 4. Pretrain + inline eval (logs eval metrics to the live wandb run before finishing).
# MLM pseudo-likelihood scoring often beats causal on BLiMP for hybrid models, so
# train.py runs both backends when --eval is not 'none'.
uv run python main.py pretrain \
    --config configs/a40.json \
    --output-dir checkpoints/ \
    --wandb \
    --max-epochs 10 \
    --eval fast

echo "All steps completed successfully at $(date)"
