#!/usr/bin/env bash
# Marathi <-> Dehwali Bhili: baseline, train, evaluate, in one job. Split across
# three jobs each would re-pull the 15.9 GB base model at ~5 min of A100 time.
set -euxo pipefail
export HF_HOME=/workspace/.hf HF_HUB_CACHE=/workspace/.hf/hub
export RESULTS_REPO="${RESULTS_REPO:-Arushhh/indic-translate-mar-bhili-lora}"
export WANDB_MODE="${WANDB_MODE:-offline}"
MAX_STEPS="${MAX_STEPS:-900}"
mkdir -p "$HF_HUB_CACHE"
df -h /workspace 2>/dev/null | tail -1 || true
nvidia-smi

git clone --depth 1 https://github.com/Bodhan-AI/bodhan_genai /workspace/tk
cd /workspace/tk
# --no-flash-attn skips a source build that MT has no use for
./install.sh --extras all-mt --no-flash-attn
source .venv/bin/activate
pip install -q sacrebleu indic-nlp-library

# preflight: fail in ~90s rather than 40 minutes in
python - <<'PY'
import torch, transformers
from packaging.version import parse
assert torch.cuda.is_available(), "no GPU visible"
print("PREFLIGHT 1/4 gpu:", torch.cuda.get_device_name(0), flush=True)
assert parse(transformers.__version__) >= parse("5.12"), transformers.__version__
print("PREFLIGHT 2/4 transformers:", transformers.__version__, flush=True)
from transformers import AutoConfig
assert AutoConfig.from_pretrained("bodhan-ai/indic-translate").model_type == "gemma4"
print("PREFLIGHT 3/4 gated model OK", flush=True)
import sacrebleu
c = sacrebleu.CHRF(word_order=2); c.corpus_score(["a"], [["a"]])
assert "nw:2" in str(c.get_signature())     # nw:2 is chrF++; CHRF() alone is chrF
print("PREFLIGHT 4/4 chrF++ OK |", c.get_signature(), flush=True)
PY

mkdir -p data/bhili configs
cp /workspace/in/data/*.jsonl data/bhili/
cp /workspace/in/configs/*.yaml configs/

# Baseline first, pushing as it goes, so a crash in training still leaves it
# banked. Bhili is not one of the 22 served languages; extra_languages names it.
python /workspace/in/eval_bhili.py --tag baseline --max-items 400

python -m bodhan_genai.mt.data.render --config configs/render_bhili.yaml --dry-run
python -m bodhan_genai.mt.data.render --config configs/render_bhili.yaml
wc -l data/mt/rendered/bhili/train.jsonl data/mt/rendered/bhili/dev.jsonl
python - "$MAX_STEPS" <<'PY'
import sys, yaml
c = yaml.safe_load(open("configs/lora_bhili.yaml"))
c["training"].update(max_steps=int(sys.argv[1]), save_steps=150, eval_steps=150,
                     save_total_limit=10, run_name="mar-bhili-v1")
c["logging"]["wandb_run_name"] = "mar-bhili-v1"
yaml.safe_dump(c, open("configs/lora_bhili.yaml", "w"))
PY
scripts/mt/train_lora.sh configs/lora_bhili.yaml

ADIR=training_output/bhili-lora
if [ ! -f "$ADIR/adapter_config.json" ]; then
  ADIR=$(ls -d training_output/bhili-lora/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
fi
[ -f "$ADIR/adapter_config.json" ] || { echo "NO ADAPTER"; exit 1; }
echo "evaluating adapter: $ADIR"
python /workspace/in/eval_bhili.py --tag finetuned --adapter "$ADIR" --max-items 400
echo "BHILI JOB COMPLETE"
