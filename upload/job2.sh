#!/usr/bin/env bash
# JOB 2 -- all three training arms, sequentially, in ONE job.
#
# Sequential-in-one-job is deliberate: each job pays ~5 min of A100 time
# pulling the 16 GB base model and waiting in queue. Three separate jobs would
# pay that three times (~$1 of pure I/O) for no benefit, since the arms do not
# need to run in parallel.
#
# MAX_STEPS is passed in, sized from the throughput measured by Job 1's probe
# rather than guessed.
set -euxo pipefail
export HF_HOME=/workspace/.hf HF_HUB_CACHE=/workspace/.hf/hub
export RESULTS_REPO="${RESULTS_REPO:-Arushhh/indic-translate-mr-lora}"
MAX_STEPS="${MAX_STEPS:-1500}"
mkdir -p "$HF_HUB_CACHE"

# Instrumentation the previous job lacked: we did not know the ephemeral disk
# ceiling, and the 16 GB model plus a torch/vLLM venv is the thing most likely
# to hit it.
df -h /workspace / 2>/dev/null || df -h
nvidia-smi

git clone --depth 1 https://github.com/Bodhan-AI/bodhan_genai /workspace/tk
cd /workspace/tk
./install.sh --extras all-mt --no-flash-attn
source .venv/bin/activate
pip install -q sacrebleu indic-nlp-library

mkdir -p data/bitext configs
cp /workspace/in/*.jsonl data/bitext/
cp /workspace/in/configs/*.yaml configs/

for ARM in samanantar bpcc shiksha; do
  echo "=== ARM: $ARM ==="
  python - "$ARM" "$MAX_STEPS" <<'PY'
import sys, yaml
arm, max_steps = sys.argv[1], int(sys.argv[2])
c = yaml.safe_load(open("configs/lora.yaml"))
c["data"]["train_file"] = f"data/mt/rendered/{arm}/train.jsonl"
c["data"]["dev_file"]   = f"data/mt/rendered/{arm}/dev.jsonl"
c["data"]["cache_dir"]  = f"data/mt/cache/{arm}"
c["training"].update(
    output_dir=f"training_output/{arm}",
    max_steps=max_steps,
    # Log-spaced checkpoints are approximated by dense saves; we EVALUATE at
    # 100/200/400/800/1600 in Job 3. Degradation on an already-converged model
    # is fastest early, so uniform spacing wastes resolution where it matters.
    save_steps=100, eval_steps=100, save_total_limit=20,
    hub_model_id=f"Arushhh/indic-translate-mr-lora-{arm}",
    hub_private_repo=False, hub_strategy="all_checkpoints",
)
yaml.safe_dump(c, open(f"configs/{arm}.yaml", "w"))
PY
  python -m bodhan_genai.mt.data.render --config "configs/render_${ARM}.yaml"
  wc -l "data/mt/rendered/${ARM}/train.jsonl" "data/mt/rendered/${ARM}/dev.jsonl"
  # One arm failing must not lose the other two -- they are already pushed.
  scripts/mt/train_lora.sh "configs/${ARM}.yaml" || echo "ARM $ARM FAILED, continuing"
  df -h /workspace 2>/dev/null | tail -1 || true
  echo "=== ARM $ARM DONE ==="
done
echo "JOB2 COMPLETE"
