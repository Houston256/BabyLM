#!/bin/bash
# Run this on your LAPTOP to download results from MetaCentrum

# Load credentials from local .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found."
    exit 1
fi

if [ -z "$METACENTRUM_USER" ]; then
    echo "Error: METACENTRUM_USER is not set in your .env file."
    exit 1
fi

# Hardcoded project defaults for the team (Praha 1)
STORAGE_HOST="tarkil.grid.cesnet.cz"
REMOTE_PATH="/storage/praha1/home/$METACENTRUM_USER/BabyLM/"

echo "Syncing checkpoints from $STORAGE_HOST..."
mkdir -p ./checkpoints
rsync -avzP "$METACENTRUM_USER@$STORAGE_HOST:${REMOTE_PATH}checkpoints/" "./checkpoints/"

echo ""
echo "Available runs:"
ls ./checkpoints/
echo ""
echo "Done! Run: uv run python chat.py <run_name>"
