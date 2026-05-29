#!/bin/bash
#PBS -N BabyLM_Full_Pipeline
#PBS -l walltime=4:00:00
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
# DATA_ARGS controls corpus preprocessing. The default reproduces our best baseline run: raw
# per-source files segmented into documents, text kept verbatim (no tag stripping — the official
# cleans nothing). Other modes:
#   HF flat (utterances):  -v DATA_ARGS=""
#   strip+utterance SEP:   -v DATA_ARGS="--strip-speaker-tags --insert-sep"
# The bin is rebuilt in-scratch every job, so DATA_ARGS is what trains.
DATA_ARGS=${DATA_ARGS:-"--source-mode raw --raw-dir data/raw"}

# Always pull the raw per-source txt files (55 MB total) so --source-mode raw works.
mkdir -p data/raw
for src in childes simple_wiki gutenberg open_subtitles bnc_spoken switchboard; do
    if [ ! -f "data/raw/${src}.train.txt" ]; then
        curl -sL "https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict-Small/resolve/main/${src}.train.txt" \
            -o "data/raw/${src}.train.txt"
    fi
done

# Tokenizer: default is the official GPT-BERT tokenizer (our best baseline). To instead train our
# own 16384 BPE, pass -v TOKENIZER=models/tokenizer.json.
TOKENIZER=${TOKENIZER:-models/gpt-bert-official.json}
if [ "$TOKENIZER" = "models/gpt-bert-official.json" ]; then
    bash scripts/fetch_tokenizer.sh "$TOKENIZER"   # download the official tokenizer from the HF Hub
fi
if [ "$TOKENIZER" = "models/tokenizer.json" ]; then
    uv run python main.py train-tokenizer --vocab-size 16384
fi
# shellcheck disable=SC2086  # intentional word splitting on DATA_ARGS
uv run python main.py tokenize-corpus --tokenizer "$TOKENIZER" --output data/train.bin $DATA_ARGS

# 4. Pretrain + inline eval (logs eval metrics to the live wandb run before finishing).
# A bare run reproduces our best baseline (defaults live in add_pretrain_args; official tokenizer +
# raw document data are set above). Override any subset via ARGS, e.g.:
#   qsub -v ARGS="--pos-emb rope --rope-base 1000" scripts/submit_babylm.sh
# Document-packing variant (targets entity_tracking):
#   qsub -v ARGS="--document-packing" scripts/submit_babylm.sh
ARGS=${ARGS:-}

# shellcheck disable=SC2086  # intentional word splitting on ARGS
uv run python main.py pretrain \
    --tokenizer "$TOKENIZER" \
    --output-dir checkpoints/ \
    --wandb \
    --eval fast \
    $ARGS

echo "All steps completed successfully at $(date)"
