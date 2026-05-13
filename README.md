# BabyLM
Neural language models semestral work (BabyLM challenge 2026)

## Commands

```bash
# 1. Tokenize: train BPE then pack the corpus into a uint16 bin
uv run python main.py train-tokenizer
uv run python main.py tokenize-corpus

# 2. Train: pretrain GPT-BERT; each save is a self-contained HF dir
uv run python main.py pretrain --config configs/small.json --wandb

# 3. Eval: run the strict-small zero-shot suite on a checkpoint dir
# One-time: unzip EWoK fast + download/filter full EWoK.
# Requires HF_TOKEN in .env and accepting terms at
# https://huggingface.co/datasets/ewok-core/ewok-core-1.0
./scripts/setup_eval_data.sh
./scripts/eval.sh checkpoints/final fast causal

# 4. Chat: interactive text completion from a checkpoint
uv run python chat.py <run_name>
# e.g. uv run python chat.py bs64_s100000_wu500_lr0.0003_mlr0.1_wd0.1_gc1.0_ga4_mp0.15_hn15_hd16
```

## MetaCentrum cluster

### First-time setup (local)

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Set credentials
cp .env.example .env
# edit .env: set METACENTRUM_USER and WANDB_API_KEY
chmod 600 .env

# Upload project (includes .env)
./scripts/upload_to_cluster.sh
```

### First-time setup (on the cluster, do once)

```bash
ssh your_username@tarkil.grid.cesnet.cz

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

echo ". \"/storage/praha1/home/$USER/.local/bin/env\"" >> ~/.profile
echo "export UV_CACHE_DIR=/storage/brno2/home/$USER/.uv_cache/" >> ~/.profile
source ~/.profile

# Verify
uv --version
```

### Running jobs

```bash
# Submit (on the cluster)
qsub BabyLM/scripts/submit_babylm.sh

# Monitor (on the cluster)
qstat -u $USER

# Download checkpoints (locally, after job finishes)
./scripts/download_results.sh
```
