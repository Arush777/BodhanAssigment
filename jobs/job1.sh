#!/usr/bin/env bash
# JOB 1 -- baseline eval of the unmodified model + a 20-step LoRA/push probe.
set -euxo pipefail
export HF_HOME=/workspace/.hf HF_HUB_CACHE=/workspace/.hf/hub
export RESULTS_REPO="${RESULTS_REPO:-Arushhh/indic-translate-mr-lora}"
mkdir -p "$HF_HUB_CACHE"
# Ephemeral-disk headroom: the 16 GB base model plus a torch/vLLM venv is the
# largest thing this job touches, so this is where a limit would bite.
df -h /workspace / 2>/dev/null || df -h
free -g || true
nproc || true
nvidia-smi || true

git clone --depth 1 https://github.com/Bodhan-AI/bodhan_genai /workspace/tk
cd /workspace/tk
# --extras all-mt, NOT the default "all": the default keeps flash-attn, which
# install.sh builds FROM SOURCE (tens of minutes) and which is a TTS-training
# dependency we can never use -- Gemma 4's 7 global layers have head_dim 512,
# above FA2's 256 cap, so the model card specifies sdpa. A first attempt at this
# job burned ~15 min of A100 time compiling it before being cancelled.
./install.sh --extras all-mt --no-flash-attn
source .venv/bin/activate
# BLEU on Devanagari is meaningless without IndicNLP pre-tokenisation.
pip install -q sacrebleu indic-nlp-library

# ---- preflight: fail in ~90s, not 40 minutes --------------------------------
python - <<'PY'
import torch, transformers
from packaging.version import parse
assert torch.cuda.is_available(), "no GPU visible -- CUDA/vLLM wheel mismatch"
print("PREFLIGHT 1/5 gpu:", torch.cuda.get_device_name(0), flush=True)
assert parse(transformers.__version__) >= parse("5.12"), transformers.__version__
print("PREFLIGHT 2/5 transformers:", transformers.__version__, flush=True)
from transformers import AutoConfig
c = AutoConfig.from_pretrained("bodhan-ai/indic-translate")
assert c.model_type == "gemma4", c.model_type
print("PREFLIGHT 3/5 gated model config OK:", c.model_type, flush=True)
from datasets import load_dataset
d = load_dataset("ai4bharat/IN22-Gen", split="test")
assert "mar_Deva" in d.column_names and len(d) == 1024, (len(d), d.column_names[:5])
print("PREFLIGHT 4/5 IN22-Gen OK:", len(d), "rows", flush=True)
import sacrebleu
# get_signature() raises until the metric has scored something -- the signature
# does not know the reference count before then. Score a dummy pair first.
chrf = sacrebleu.CHRF(word_order=2)
chrf.corpus_score(["a"], [["a"]])
sig = str(chrf.get_signature())
assert "nw:2" in sig, sig        # nw:2 == chrF++; CHRF() alone defaults to nw:0
print("PREFLIGHT 5/5 chrF++ signature:", sig, flush=True)
print("PREFLIGHT OK", flush=True)
PY

cp /workspace/in/*.jsonl data/bitext/ 2>/dev/null || { mkdir -p data/bitext && cp /workspace/in/*.jsonl data/bitext/; }
mkdir -p configs && cp /workspace/in/configs/*.yaml configs/

# ---- baseline evaluation (pushes artifacts as they land) --------------------
python /workspace/in/eval_baseline.py --tag baseline --mar-samples 1024 --probe-samples 300

# ---- 20-step probe: validates peft target resolution + hub push -------------
python - <<'PY'
import yaml
c = yaml.safe_load(open("configs/lora.yaml"))
c["data"]["train_file"] = "data/mt/rendered/shiksha/train.jsonl"
c["data"]["dev_file"]   = "data/mt/rendered/shiksha/dev.jsonl"
c["training"].update(output_dir="training_output/probe", max_steps=6,
                     save_steps=6, eval_steps=6, save_total_limit=1,
                     hub_model_id="Arushhh/indic-translate-mr-lora",
                     hub_private_repo=False)
yaml.safe_dump(c, open("configs/probe.yaml", "w"))
PY
python -m bodhan_genai.mt.data.render --config configs/render_shiksha.yaml
scripts/mt/train_lora.sh configs/probe.yaml
echo "JOB1 COMPLETE"
