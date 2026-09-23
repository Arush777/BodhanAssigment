#!/usr/bin/env bash
# JOB 1 -- baseline evaluation of the UNMODIFIED model, plus a 20-step
# training probe that validates the LoRA + Hub-push path.
#
# Everything downstream is a delta against these numbers, so this runs first.
# The training probe at the end is deliberate: a baseline eval alone never
# exercises peft target-module resolution or push_to_hub, and those are the two
# failure modes that would otherwise surface 40 minutes into the real run.
set -euo pipefail

TIMEOUT="${TIMEOUT:-50m}"
FLAVOR="${FLAVOR:-a100-large}"
: "${HF_TOKEN:?export HF_TOKEN first}"

read -r -d '' JOB <<'INNER' || true
set -euxo pipefail
export HF_HOME=/workspace/.hf HF_HUB_CACHE=/workspace/.hf/hub
mkdir -p "$HF_HUB_CACHE"
nvidia-smi

git clone --depth 1 https://github.com/Bodhan-AI/bodhan_genai /workspace/tk
cd /workspace/tk && ./install.sh && source .venv/bin/activate

# ---- preflight: fail in ~90s, not 40min -------------------------------------
python - <<'PY'
import torch, transformers
from packaging.version import parse
assert torch.cuda.is_available(), "no GPU -- CUDA/vLLM wheel mismatch"
assert parse(transformers.__version__) >= parse("5.12"), transformers.__version__
from transformers import AutoConfig
c = AutoConfig.from_pretrained("bodhan-ai/indic-translate")     # gated -> proves token
assert c.model_type == "gemma4"
import datasets; datasets.load_dataset("ai4bharat/IN22-Gen", split="test[:2]")  # gated -> proves eval access
print("PREFLIGHT OK", torch.cuda.get_device_name(0), transformers.__version__)
PY

hf download Arushhh/bodhan-mt-marathi-data --repo-type dataset --local-dir /workspace/in
mkdir -p data/bitext configs && cp /workspace/in/*.jsonl data/bitext/ && cp /workspace/in/configs/*.yaml configs/

# ---- baseline generations + metrics -----------------------------------------
python /workspace/in/eval_baseline.py 2>&1 | tail -40

# ---- 20-step training probe: validates peft targets + hub push --------------
python - <<'PY'
import yaml, pathlib
c = yaml.safe_load(open("configs/lora.yaml"))
c["data"]["train_file"] = "data/mt/rendered/shiksha/train.jsonl"
c["data"]["dev_file"]   = "data/mt/rendered/shiksha/dev.jsonl"
t = c["training"]
t.update(output_dir="training_output/probe", max_steps=20, num_train_epochs=1,
         save_steps=20, eval_steps=20, save_total_limit=1,
         hub_model_id="Arushhh/indic-translate-mr-probe")
yaml.safe_dump(c, open("configs/probe.yaml","w"))
PY
python -m bodhan_genai.mt.data.render --config configs/render_shiksha.yaml
scripts/mt/train_lora.sh configs/probe.yaml 2>&1 | tail -25
echo "JOB1 COMPLETE"
INNER

echo "flavor=$FLAVOR timeout=$TIMEOUT  est \$$(python3 -c "print(f'{2.5*50/60:.2f}')")"
hf jobs run --flavor "$FLAVOR" --timeout "$TIMEOUT" --secrets HF_TOKEN \
  huggingface/trl bash -lc "$JOB"
