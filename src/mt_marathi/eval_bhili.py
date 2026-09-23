"""Evaluate Marathi<->Bhili translation with diagnostics that isolate the two
reported failure modes.

Global chrF++ cannot answer either question the hiring team asked. Numeral
errors affect 44.6% of sentences but move the corpus average only slightly, and
short phrases are 9% of the data. Worse, on numerals a model that simply COPIES
its input preserves them perfectly -- so "did the numbers survive" scores 100%
for the degenerate solution and cannot show improvement at all.

So this reports, per test set and per direction:

  chrF++ (word_order=2)   general quality; CHRF() defaults to 0, i.e. plain
                          chrF, so word_order is passed explicitly
  NSEM                    numeral sequence exact match, digit-script normalised
  digit-script match      whether output script follows the reference
  exact match             the only sane metric at 1-3 words
  spurious copy rate      among items where reference != source, how often the
                          hypothesis is just the source echoed back. This is the
                          metric that actually names the short-phrase complaint.

Every number is reported against a COPY baseline and against the REFERENCE's own
numeral-preservation rate (measured at 83.7% on this corpus), because without
those two rows none of the numbers mean anything.
"""

from __future__ import annotations

import argparse, json, os, re, time
from pathlib import Path

DEVA = "०१२३४५६७८९"
TO_ASCII = str.maketrans(DEVA, "0123456789")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
RESULTS_REPO = os.environ.get("RESULTS_REPO", "Arushhh/indic-translate-mar-bhili-lora")


def register_bhili() -> None:
    """Teach the inference engine a language the served model does not have.

    `extra_languages` in the render config extends the TRAINING prompt, but the
    inference engine keeps its own frozen table and raises on anything outside
    the served 25:

        ValueError: unsupported language 'Bhili'.

    So the toolkit can train a new language but cannot serve it. Patching the
    same two module-level dicts `resolve_language` consults is the minimal fix,
    and it is the exact analogue of `extra_languages` -- which matters, because
    the prompt rendered at inference must match the one rendered in training or
    the model sees a string it was never trained on.
    """
    from bodhan_genai.mt.templates import prompt as _p
    _p.LANGUAGE_NAMES["bhb_Deva"] = "Bhili"
    _p._BY_NAME["bhili"] = "Bhili"
    assert _p.resolve_language("Bhili") == "Bhili"
    assert _p.resolve_language("bhb_Deva") == "Bhili"


def numerals(t): return NUM_RE.findall(t.translate(TO_ASCII))
def deva_digits(t): return sum(c in DEVA for c in t)
def norm(t): return re.sub(r"\s+", " ", t.strip().strip(".।,"))


def push(api, local, path_in_repo):
    for i in range(3):
        try:
            api.upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo,
                            repo_id=RESULTS_REPO, repo_type="model")
            print(f"    pushed -> {path_in_repo}", flush=True); return
        except Exception as e:
            print(f"    push retry {i+1}: {e}", flush=True); time.sleep(5 * (i + 1))


def score(preds, srcs, refs):
    import sacrebleu
    chrf = sacrebleu.CHRF(word_order=2)          # word_order=2 IS chrF++
    out = {"n": len(preds), "chrf2": round(chrf.corpus_score(preds, [refs]).score, 3)}

    withnum = [(p, s, r) for p, s, r in zip(preds, srcs, refs) if numerals(s)]
    if withnum:
        out["nsem"] = round(sum(numerals(p) == numerals(s) for p, s, _ in withnum) / len(withnum) * 100, 1)
        out["nsem_vs_ref"] = round(sum(numerals(p) == numerals(r) for p, _, r in withnum) / len(withnum) * 100, 1)
        out["ref_preserves"] = round(sum(numerals(s) == numerals(r) for _, s, r in withnum) / len(withnum) * 100, 1)
        script_ok = sum((deva_digits(p) > 0) == (deva_digits(r) > 0) for p, _, r in withnum)
        out["digit_script_match"] = round(script_ok / len(withnum) * 100, 1)
        out["n_numeral"] = len(withnum)

    out["exact"] = round(sum(norm(p) == norm(r) for p, r in zip(preds, refs)) / len(preds) * 100, 1)
    hard = [(p, s) for p, s, r in zip(preds, srcs, refs) if norm(s) != norm(r)]
    if hard:
        out["spurious_copy"] = round(sum(norm(p) == norm(s) for p, s in hard) / len(hard) * 100, 1)
        out["n_hard"] = len(hard)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--sets", nargs="+", default=["GENERAL", "NUM_NAT", "SHORT_NAT", "SHORT_HARD"])
    ap.add_argument("--data-dir", type=Path, default=Path("data/bhili"))
    ap.add_argument("--max-items", type=int, default=400)
    args = ap.parse_args()

    from huggingface_hub import HfApi
    from bodhan_genai.mt import IndicMTEngine
    register_bhili()
    print("  registered bhb_Deva -> 'Bhili' with the inference engine", flush=True)
    api, out = HfApi(), Path("results"); out.mkdir(exist_ok=True)
    # upload_file does NOT create a missing repo -- it fails, retries, and gives
    # up silently. An earlier run computed a full baseline and lost every number
    # to exactly this. Create first, always.
    api.create_repo(RESULTS_REPO, repo_type="model", exist_ok=True)
    print(f"  results repo ready: {RESULTS_REPO}", flush=True)
    metrics = {}

    engine = IndicMTEngine("bodhan-ai/indic-translate", backend="hf",
                           **({"adapter_dir": args.adapter} if args.adapter else {}))
    with engine:
        for name in args.sets:
            path = args.data_dir / f"eval_{name}.jsonl"
            if not path.exists():
                print(f"  skip {name} (missing)"); continue
            rows = [json.loads(l) for l in path.open(encoding="utf-8")][: args.max_items]
            mar = [r["mar"] for r in rows]; bhb = [r["bhb"] for r in rows]

            for direction, srcs, refs, tgt in (("mar-bhb", mar, bhb, "Bhili"),
                                               ("bhb-mar", bhb, mar, "Marathi")):
                key = f"{name}.{direction}"
                print(f"  [{args.tag}] {key}  n={len(srcs)}", flush=True)
                preds = [(x.text or "") for x in engine.translate_batch(srcs, tgt_lang=tgt)]

                gp = out / f"{args.tag}.{key}.jsonl"
                with gp.open("w", encoding="utf-8") as fh:
                    for s, p, r in zip(srcs, preds, refs):
                        fh.write(json.dumps({"src": s, "pred": p, "ref": r}, ensure_ascii=False) + "\n")
                push(api, gp, f"generations/{gp.name}")       # raw output first

                metrics[key] = score(preds, srcs, refs)
                metrics[f"{key}.COPY"] = score(srcs, srcs, refs)   # the baseline that matters
                print(f"    {metrics[key]}", flush=True)
                mp = out / f"{args.tag}.metrics.json"
                mp.write_text(json.dumps(metrics, indent=2))
                push(api, mp, f"metrics/{mp.name}")
    print(f"\n  {args.tag} done: {len(metrics)} rows", flush=True)


if __name__ == "__main__":
    main()
