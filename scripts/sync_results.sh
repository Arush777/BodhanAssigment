#!/usr/bin/env bash
# Snapshot curves, metrics, checkpoints and a de-noised log to GitHub every
# 10 minutes while jobs are in flight.
set -uo pipefail
cd /Users/ashoksharma/Bodhan_assigment/bodhan-mt-marathi
set -a; . ../.env; set +a
. .venv/bin/activate
mkdir -p results logs

for i in $(seq 1 60); do
  PYTHONPATH=src python -m mt_marathi.curves data/job2.log --out results/ >/dev/null 2>&1

  # progress bars overwrite themselves with \r, so split on it before filtering
  tr '\r' '\n' < data/job2.log \
    | grep -vE "^\s*(Downloading|Reconstructing|Fetching|Loading weights|Tokenizing|Downloading bytes)" \
    | grep -vE "^\s*$" | tail -3000 > logs/job2_training.log 2>/dev/null

  for ARM in bhili; do
    curl -s --max-time 20 -H "Authorization: Bearer $HF_TOKEN" \
      "https://huggingface.co/api/models/Arushhh/indic-translate-mar-bhili-lora" \
      | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except: sys.exit()
if d.get('error'): sys.exit()
ck=sorted({s['rfilename'].split('/')[0] for s in d.get('siblings',[]) if s['rfilename'].startswith('checkpoint-')},key=lambda x:int(x.split('-')[1]))
print(json.dumps({'repo':d.get('id'),'checkpoints':ck}, indent=2))" > "results/checkpoints_$ARM.json" 2>/dev/null
  done
  curl -sL --max-time 25 -H "Authorization: Bearer $HF_TOKEN" \
    "https://huggingface.co/Arushhh/indic-translate-mr-lora/resolve/main/metrics/baseline.metrics.json" \
    -o results/baseline.metrics.json 2>/dev/null

  if [ -n "$(git status --porcelain results logs)" ]; then  # skip empty commits
    git add results logs >/dev/null 2>&1
    git -c user.email=23je0154@iitism.ac.in -c user.name="Arush Sharma" \
        commit -q -m "Sync results: curves, metrics, checkpoints ($(date '+%H:%M'))" >/dev/null 2>&1
    git remote set-url origin "https://x-access-token:${GITHUB_TOKEN}@github.com/Arush777/BodhanAssigment.git"
    git push -q origin main >/dev/null 2>&1 && echo "$(date '+%H:%M') pushed $(git rev-parse --short HEAD)"
    git remote set-url origin "https://github.com/Arush777/BodhanAssigment.git"
  fi
  sleep 600
done
