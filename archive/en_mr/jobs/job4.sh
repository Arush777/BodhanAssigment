#!/usr/bin/env bash
# JOB 4 -- merge the chosen arm's adapter into a self-contained, servable
# checkpoint and prove it loads under vLLM.
#
# WINNER_ARM is passed in after Job 3's numbers are known, so the deliverable is
# the arm the evidence selects rather than one chosen in advance.
#
# The merge is the one path not yet exercised. Four things must happen and all
# four are required for the result to load (docs/mt/end-to-end.md §3):
#   1. fold W + (alpha/r)·B@A into the base weights
#   2. re-enable use_cache (training turns it off)
#   3. stage tokenizer AND processor_config.json from the base
#   4. add the KV-shared k_norm sidecar -- Gemma 4 E4B shares K/V across its
#      last 18 decoder layers and stores no k_norm for them, while vLLM builds
#      the module for every layer and its loader aborts without it
# scripts/mt/merge.sh does all four; a hand-rolled merge_and_unload() does not.
set -euxo pipefail
export HF_HOME=/workspace/.hf HF_HUB_CACHE=/workspace/.hf/hub
WINNER_ARM="${WINNER_ARM:-shiksha}"
OUT_REPO="${OUT_REPO:-Arushhh/indic-translate-mr-merged}"
mkdir -p "$HF_HUB_CACHE"
df -h /workspace 2>/dev/null | tail -1 || true
nvidia-smi

git clone --depth 1 https://github.com/Bodhan-AI/bodhan_genai /workspace/tk
cd /workspace/tk
./install.sh --extras all-mt --no-flash-attn
source .venv/bin/activate

# Resolve the adapter: prefer the repo root, else the highest checkpoint.
hf download "Arushhh/indic-translate-mr-lora-${WINNER_ARM}" --local-dir /workspace/ad
ADIR=/workspace/ad
if [ ! -f "$ADIR/adapter_config.json" ]; then
  ADIR=$(ls -d /workspace/ad/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
fi
[ -f "$ADIR/adapter_config.json" ] || { echo "NO ADAPTER FOUND"; exit 1; }
echo "merging adapter: $ADIR"

# Record what the adapter actually targeted -- this is also the artifact that
# settles the 23.2M gap between the estimated and reported trainable count.
python -c "import json;d=json.load(open('$ADIR/adapter_config.json'));print(json.dumps(d,indent=2))"

scripts/mt/merge.sh "$ADIR" /workspace/merged

# Prove it loads under vLLM rather than assuming the sidecar worked.
python - <<'PY'
from vllm import LLM, SamplingParams
llm = LLM(model="/workspace/merged", max_model_len=2048, gpu_memory_utilization=0.85)
prompt = ("<bos><|turn>user\n"
          "Translate the following text into Marathi:\n\n"
          "The committee approved the proposal.<turn|>\n<|turn>model\n")
out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=128, stop=["<turn|>"]))
print("VLLM LOAD OK ->", out[0].outputs[0].text.strip())
PY

hf upload "$OUT_REPO" /workspace/merged . --repo-type model \
  --commit-message "Merged ${WINNER_ARM} LoRA into servable checkpoint"
echo "JOB4 COMPLETE"
