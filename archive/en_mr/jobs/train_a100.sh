#!/usr/bin/env bash
# Launch the LoRA finetune on HF Jobs (1x A100 80GB, $2.50/hr).
#
# Two facts about HF Jobs drive this script's shape:
#   1. Disk is EPHEMERAL. Anything not pushed to the Hub during the run is lost,
#      so the trainer is configured with hub_strategy=all_checkpoints.
#   2. Default timeout is 30 MINUTES, which silently killed three earlier jobs
#      on this account. --timeout is therefore never omitted.
#
# Usage:  ./jobs/train_a100.sh [timeout]
set -euo pipefail

TIMEOUT="${1:-3h}"
FLAVOR="${FLAVOR:-a100-large}"
DATA_REPO="${DATA_REPO:-Arushhh/bodhan-mt-marathi-data}"

: "${HF_TOKEN:?set HF_TOKEN (source ../.env) before launching}"

# The whole job body. Runs non-interactively; every stage fails loudly.
read -r -d '' JOB_SCRIPT <<'INNER' || true
set -euxo pipefail

export HF_HOME=/workspace/.hf
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_HUB_CACHE"

# ---------------------------------------------------------------- preflight --
# ~90 seconds, and it turns three classes of silent failure into an immediate
# exit. Cheap insurance at $2.50/hr.
nvidia-smi
git clone --depth 1 https://github.com/Bodhan-AI/bodhan_genai /workspace/toolkit
cd /workspace/toolkit
./install.sh
source .venv/bin/activate

python - <<'PY'
import torch, transformers
from packaging.version import parse
assert torch.cuda.is_available(), "no GPU visible -- CUDA/vLLM wheel mismatch"
assert parse(transformers.__version__) >= parse("5.12"), f"transformers {transformers.__version__} < 5.12"
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("bodhan-ai/indic-translate")   # gated: proves the token works
assert cfg.model_type == "gemma4", cfg.model_type
print("PREFLIGHT OK:", torch.cuda.get_device_name(0), "| transformers", transformers.__version__)
PY

# ------------------------------------------------------------------- data --
hf download "$DATA_REPO" --repo-type dataset --local-dir /workspace/inputs
mkdir -p data/bitext configs
cp /workspace/inputs/samanantar_en_mr.jsonl data/bitext/
cp /workspace/inputs/render.yaml /workspace/inputs/lora.yaml configs/

python -m bodhan_genai.mt.data.render --config configs/render.yaml --dry-run
python -m bodhan_genai.mt.data.render --config configs/render.yaml
wc -l data/mt/rendered/train.jsonl data/mt/rendered/dev.jsonl

# ------------------------------------------------------------------ train --
scripts/mt/train_lora.sh configs/lora.yaml
INNER

echo "launching: flavor=$FLAVOR timeout=$TIMEOUT"
echo "estimated cost at \$2.50/hr: ~\$$(python3 -c "print(f'{2.50*float('${TIMEOUT%h}'):.2f}')")"

hf jobs run \
  --flavor "$FLAVOR" \
  --timeout "$TIMEOUT" \
  --secrets HF_TOKEN \
  -e DATA_REPO="$DATA_REPO" \
  huggingface/trl \
  bash -lc "$JOB_SCRIPT"
