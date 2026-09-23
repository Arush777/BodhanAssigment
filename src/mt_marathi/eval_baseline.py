"""Evaluate indic-translate (base or LoRA-adapted) and push results as they land.

Design constraints this file exists to satisfy:

*   HF Job disk is EPHEMERAL. Every artifact is uploaded to the Hub the moment
    it is produced, never batched to the end -- a timeout must cost time, not
    work.
*   RAW GENERATIONS ARE THE POINT. Metrics can be recomputed from generations;
    generations cannot be recovered from metrics. Without them, every follow-up
    question ("which sentences got worse?") costs another GPU job.
*   The prompt contract is owned by bodhan_genai, not by this file. We call
    IndicMTEngine rather than formatting prompts ourselves, because a wrong
    prompt produces fluent, quietly worse output and no error.

Usage:
    python eval_baseline.py --tag baseline
    python eval_baseline.py --tag shiksha-final --adapter training_output/shiksha/checkpoint-1500
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

RESULTS_REPO = os.environ.get("RESULTS_REPO", "Arushhh/indic-translate-mr-lora")

# Cross-lingual probe: does Marathi-only finetuning damage other languages?
# 2x2 on {script x family} with Marathi as the treated cell.
#   hin_Deva  same script, same family   (Indo-Aryan, Devanagari)
#   guj_Gujr  diff script, same family
#   tam_Taml  diff script, diff family   (Dravidian) -- control
PROBE_LANGS = ["mar_Deva", "hin_Deva", "guj_Gujr", "tam_Taml"]


def push(api, local: Path, path_in_repo: str) -> None:
    """Upload immediately; never let an artifact exist only on job disk."""
    for attempt in range(3):
        try:
            api.upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo,
                            repo_id=RESULTS_REPO, repo_type="model")
            print(f"    pushed -> {path_in_repo}", flush=True)
            return
        except Exception as exc:                       # transient Hub errors
            print(f"    push failed ({attempt+1}/3): {exc}", flush=True)
            time.sleep(5 * (attempt + 1))
    print(f"    GIVING UP on {path_in_repo}", flush=True)


def indic_tokenize(lines: list[str], lang: str) -> list[str]:
    """IndicNLP tokenisation, required before BLEU on Devanagari targets.

    sacrebleu's default 13a tokenizer is built for Latin script. IndicTrans2's
    compute_metrics.sh normalises + tokenises with IndicNLP and then passes
    tok='none'. Passing tok='none' on RAW Devanagari instead gives whitespace
    BLEU with punctuation glued to words -- a silently meaningless number.
    """
    try:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        from indicnlp.tokenize import indic_tokenize
    except ImportError:
        print("    WARNING: indic-nlp-library missing; BLEU for this direction is NOT comparable")
        return lines
    normalizer = IndicNormalizerFactory().get_normalizer(lang.split("_")[0])
    return [" ".join(indic_tokenize.trivial_tokenize(normalizer.normalize(l), lang.split("_")[0]))
            for l in lines]


def score(preds: list[str], refs: list[str], tgt_lang: str) -> dict:
    import sacrebleu

    # chrF++ is chrF with word_order=2. sacrebleu's CHRF() DEFAULTS TO 0,
    # i.e. plain chrF -- passing this explicitly is not optional.
    chrf = sacrebleu.CHRF(word_order=2)
    chrf_score = chrf.corpus_score(preds, [refs])

    if tgt_lang == "eng_Latn":
        bleu = sacrebleu.BLEU()                       # 13a, correct for English
        bleu_score = bleu.corpus_score(preds, [refs])
    else:
        p, r = indic_tokenize(preds, tgt_lang), indic_tokenize(refs, tgt_lang)
        bleu = sacrebleu.BLEU(tokenize="none")
        bleu_score = bleu.corpus_score(p, [r])

    # Bootstrap CI over the TEST SET. This is the noise that matters when
    # comparing two models; it is far wider than decode-repeat noise.
    rng = random.Random(1234)
    n, boot = len(preds), []
    for _ in range(200):
        idx = [rng.randrange(n) for _ in range(n)]
        boot.append(chrf.corpus_score([preds[i] for i in idx], [[refs[i] for i in idx]]).score)
    boot.sort()

    # Forgetting often shows up as DEGENERATION (empty or runaway output)
    # rather than gradual decline, and that pattern is invisible in chrF++
    # alone. Pair each pred with its own ref by position -- an earlier version
    # used preds.index(p), which is O(n^2) and silently mis-pairs duplicates.
    degenerate = sum(
        1 for p, r in zip(preds, refs)
        if not p.strip() or len(p) > 12 * max(len(r), 1)
    )

    return {
        "chrf2": round(chrf_score.score, 3),
        "chrf2_ci95": [round(boot[5], 3), round(boot[194], 3)],
        "chrf2_signature": str(chrf.get_signature()),
        "bleu": round(bleu_score.score, 3),
        "bleu_signature": str(bleu.get_signature()),
        "n": n,
        "degenerate": degenerate,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="name for this evaluation run")
    ap.add_argument("--adapter", default=None, help="unmerged PEFT adapter dir")
    ap.add_argument("--mar-samples", type=int, default=1024, help="full IN22-Gen for Marathi")
    ap.add_argument("--probe-samples", type=int, default=300, help="subsample for other languages")
    ap.add_argument("--indomain-file", default=None, action="append",
                    help="NAME=PATH of a held-out in-domain set; repeatable. "
                         "Every arm is scored on EVERY in-domain set, so no arm is "
                         "tested only on its own corpus while others are not.")
    ap.add_argument("--indomain-samples", type=int, default=500)
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi
    from bodhan_genai.mt import IndicMTEngine

    api = HfApi()
    out = Path("results"); out.mkdir(exist_ok=True)

    print(f"  loading IN22-Gen", flush=True)
    ds = load_dataset("ai4bharat/IN22-Gen", split="test")
    print(f"    {len(ds)} rows, columns include {PROBE_LANGS}", flush=True)

    engine = IndicMTEngine("bodhan-ai/indic-translate", backend="hf",
                           **({"adapter_dir": args.adapter} if args.adapter else {}))

    all_metrics: dict[str, dict] = {}
    with engine:
        for lang in PROBE_LANGS:
            n = args.mar_samples if lang == "mar_Deva" else args.probe_samples
            # n<=0 means "skip this language" (used by the degradation curve,
            # which only needs Marathi). Without this guard ds.select(range(0))
            # yields an empty corpus and score() divides by zero.
            if n <= 0:
                continue
            rows = ds.select(range(min(n, len(ds))))
            eng_side, ind_side = list(rows["eng_Latn"]), list(rows[lang])

            for direction, src, refs, tgt in (
                (f"en-{lang}", eng_side, ind_side, lang),
                (f"{lang}-en", ind_side, eng_side, "eng_Latn"),
            ):
                print(f"  [{args.tag}] {direction}  n={len(src)}", flush=True)
                results = engine.translate_batch(src, tgt_lang=tgt)
                preds = [(r.text or "") for r in results]

                gen_path = out / f"{args.tag}.{direction}.jsonl"
                with gen_path.open("w", encoding="utf-8") as fh:
                    for s, p, r in zip(src, preds, refs):
                        fh.write(json.dumps({"src": s, "pred": p, "ref": r}, ensure_ascii=False) + "\n")
                push(api, gen_path, f"generations/{gen_path.name}")   # push BEFORE scoring

                m = score(preds, refs, tgt)
                all_metrics[direction] = m
                print(f"    chrF++ {m['chrf2']} CI{m['chrf2_ci95']}  BLEU {m['bleu']}", flush=True)

                mpath = out / f"{args.tag}.metrics.json"
                mpath.write_text(json.dumps(all_metrics, indent=2))
                push(api, mpath, f"metrics/{mpath.name}")             # push after every direction

        # In-domain evaluation. Without this, an arm trained on education data is
        # judged only on general-domain IN22-Gen -- we would measure the COST of
        # domain adaptation and never its BENEFIT. The held-out set is split by
        # whole course_id, so no lecture straddles train and test.
        for spec in (args.indomain_file or []):
            name, _, path = spec.partition("=")
            rows = [json.loads(l) for l in open(path, encoding="utf-8")][:args.indomain_samples]
            eng_side = [r["eng"] for r in rows]
            mar_side = [r["mar"] for r in rows]
            for direction, src, refs, tgt in (
                (f"{name}-en-mar_Deva", eng_side, mar_side, "mar_Deva"),
                (f"{name}-mar_Deva-en", mar_side, eng_side, "eng_Latn"),
            ):
                print(f"  [{args.tag}] {direction}  n={len(src)}", flush=True)
                results = engine.translate_batch(src, tgt_lang=tgt)
                preds = [(r.text or "") for r in results]

                gen_path = out / f"{args.tag}.{direction}.jsonl"
                with gen_path.open("w", encoding="utf-8") as fh:
                    for a, b, c in zip(src, preds, refs):
                        fh.write(json.dumps({"src": a, "pred": b, "ref": c}, ensure_ascii=False) + "\n")
                push(api, gen_path, f"generations/{gen_path.name}")

                m = score(preds, refs, tgt)
                all_metrics[direction] = m
                print(f"    chrF++ {m['chrf2']} CI{m['chrf2_ci95']}  BLEU {m['bleu']}", flush=True)
                mpath = out / f"{args.tag}.metrics.json"
                mpath.write_text(json.dumps(all_metrics, indent=2))
                push(api, mpath, f"metrics/{mpath.name}")

    print(f"\n  {args.tag} complete: {len(all_metrics)} directions", flush=True)


if __name__ == "__main__":
    main()
