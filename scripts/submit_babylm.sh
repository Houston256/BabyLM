#!/bin/bash
#PBS -N BabyLM_Full_Pipeline
#PBS -l walltime=0:30:00
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
    if [ -d "$SCRATCHDIR/$PROJECT_NAME/eval/strict/results" ]; then
        echo "Syncing eval results back to $DATADIR/results_$START_TIME ..."
        mkdir -p "$DATADIR/results_$START_TIME"
        rsync -av "$SCRATCHDIR/$PROJECT_NAME/eval/strict/results/" "$DATADIR/results_$START_TIME/"
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
    --exclude "checkpoints/" \
    --exclude "data/*.bin" \
    "$DATADIR/" "$SCRATCHDIR/$PROJECT_NAME/"

cd "$SCRATCHDIR/$PROJECT_NAME"

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

# 4. Pretrain
uv run python main.py pretrain \
    --config configs/a40.json \
    --batch-size 64 \
    --max-steps 10000000 \
    --output-dir checkpoints/ \
    --wandb

# 5. Evaluation
CKPT_DIR=$(find checkpoints/ -mindepth 1 -maxdepth 1 -type d | head -1)
bash scripts/eval.sh "$CKPT_DIR" fast causal

echo "All steps completed successfully at $(date)"
