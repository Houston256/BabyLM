#!/bin/bash
# Fetch the official GPT-BERT strict-small tokenizer into models/
# Source: the released BabyLM-2025 mixed baseline on the HF Hub.
set -e

DEST="${1:-models/gpt-bert-official.json}"
URL="https://huggingface.co/BabyLM-community/babylm-baseline-10m-gpt-bert-mixed/resolve/main/tokenizer.json"

if [ -f "$DEST" ]; then
    echo "tokenizer already present: $DEST"
    exit 0
fi

mkdir -p "$(dirname "$DEST")"
echo "downloading official tokenizer -> $DEST"
curl -fsSL "$URL" -o "$DEST"
echo "done ($(wc -c < "$DEST") bytes)"
