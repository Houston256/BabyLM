#!/bin/bash

# Load credentials from local .env file
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "Error: .env file not found. Please copy .env.example to .env and fill in your details."
    exit 1
fi

if [ -z "$METACENTRUM_USER" ]; then
    echo "Error: METACENTRUM_USER is not set in your .env file."
    exit 1
fi

# Hardcoded project defaults for the team (Praha 1)
STORAGE_HOST="storage-praha1.metacentrum.cz"
REMOTE_PATH=~"/BabyLM/"

echo "Syncing project to $STORAGE_HOST as $METACENTRUM_USER..."

#ssh "$METACENTRUM_USER@$STORAGE_HOST" "mkdir -p $REMOTE_PATH"

rsync -avzP \
    --exclude='.venv/' \
    --exclude='.git/' \
    --exclude='wandb/' \
    --exclude='checkpoints/' \
    --exclude='data/' \
    --exclude='models/' \
    --exclude='results/' \
    --exclude='eval/multilingual/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    ./ "$METACENTRUM_USER@$STORAGE_HOST:$REMOTE_PATH"

echo "Done! You can now log into MetaCentrum and run: qsub BabyLM/scripts/submit_babylm.sh"
