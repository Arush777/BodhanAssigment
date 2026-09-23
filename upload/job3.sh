#!/usr/bin/env bash
# JOB 3 -- one eval job covering every condition.
#
# Folded into a single job because each job pays ~5 min of A100 time pulling the
# 16 GB base model. Six separate eval jobs would spend ~$1 on I/O alone.
#
# Adapters are evaluated UNMERGED on the `hf` backend. Merging each checkpoint
# would cost time and would hit the Gemma 4 k_norm sidecar issue (the last 18 of
# 42 layers share KV and store no k_norm, so vLLM's loader aborts without a
# zeroed sidecar). We merge exactly once, at the end, in Job 4.
set -euxo pipefail
export HF_HOME=/workspace/.hf HF_HUB_CACHE=/workspace/.hf/hub
export RESULTS_REPO="${RESULTS_REPO:-Arushhh/indic-translate-mr-lora}"
mkdir -p "$HF_HUB_CACHE"
df -h /workspace 2>/dev/null | tail -1 || true
nvidia-smi

git clone --depth 1 https://github.com/Bodhan-AI/bodhan_genai /workspace/tk
cd /workspace/tk
./install.sh --extras all-mt --no-flash-attn
source .venv/bin/activate
pip install -q sacrebleu indic-nlp-library
mkdir -p data/bitext && cp /workspace/in/*.jsonl data/bitext/

# Final adapter from each arm -> the quality (A vs B) and domain (B vs C) axes.
for ARM in samanantar bpcc shiksha; do
  REPO="Arushhh/indic-translate-mr-lora-${ARM}"
  hf download "$REPO" --local-dir "/workspace/ad/${ARM}" || { echo "MISSING $REPO"; continue; }
  python /workspace/in/eval_baseline.py --tag "final-${ARM}" \
      --adapter "/workspace/ad/${ARM}" --mar-samples 300 --probe-samples 300 \
      --indomain-file edu=data/bitext/shiksha_heldout_en_mr.jsonl \
      --indomain-file news=data/bitext/bpcc_heldout_en_mr.jsonl \
      --indomain-samples 500 \
      || echo "EVAL FAILED for ${ARM}, continuing"
done

# Degradation curve: log-spaced checkpoints from the education arm only.
# Damage to an already-converged model is fastest in the first few hundred
# steps, so 100/200/400/800 resolves the knee that uniform spacing would miss.
for STEP in 100 200 400 800; do
  CKPT="/workspace/ad/shiksha/checkpoint-${STEP}"
  [ -d "$CKPT" ] || { echo "no checkpoint-${STEP}, skipping"; continue; }
  python /workspace/in/eval_baseline.py --tag "curve-shiksha-${STEP}" \
      --adapter "$CKPT" --mar-samples 300 --probe-samples 0 \
      || echo "CURVE ${STEP} FAILED, continuing"
done
echo "JOB3 COMPLETE"
