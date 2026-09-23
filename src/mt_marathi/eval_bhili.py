"""Evaluate Marathi/Bhili translation.

Global chrF++ answers neither of the reported failures. Numerals touch 45% of
sentences but barely move a corpus average, and a model that copies its input
preserves them perfectly, so numeral-preservation scores 100% for the
degenerate solution. Every row here is therefore paired with a COPY baseline,
and the metrics that carry signal are digit-script match and spurious copy rate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

DEVA = "०१२३४५६७८९"
TO_ASCII = str.maketrans(DEVA, "0123456789")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
RESULTS_REPO = os.environ.get("RESULTS_REPO", "Arushhh/indic-translate-mar-bhili-lora")


def numerals(t: str) -> list[str]:
    return NUM_RE.findall(t.translate(TO_ASCII))


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().strip(".।,"))


def register_bhili() -> None:
    """Add bhb_Deva to the inference engine's language table.

    extra_languages covers the renderer only; IndicMTEngine keeps its own table
    and raises on anything outside the served 25. The prompt built at inference
    has to match the one trained on, so both dicts get the same entry.
    """
    from bodhan_genai.mt.templates import prompt

    prompt.LANGUAGE_NAMES["bhb_Deva"] = "Bhili"
    prompt._BY_NAME["bhili"] = "Bhili"
    assert prompt.resolve_language("bhb_Deva") == "Bhili"


def push(api, local: Path, path_in_repo: str) -> None:
    for attempt in range(3):
        try:
            api.upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo,
                            repo_id=RESULTS_REPO, repo_type="model")
            return
        except Exception as exc:
            print(f"    push retry {attempt + 1}: {exc}", flush=True)
            time.sleep(5 * (attempt + 1))
    print(f"    lost {path_in_repo}", flush=True)


def score(preds: list[str], srcs: list[str], refs: list[str]) -> dict:
    import sacrebleu

    chrf = sacrebleu.CHRF(word_order=2)
    out = {"n": len(preds), "chrf2": round(chrf.corpus_score(preds, [refs]).score, 3)}

    withnum = [(p, s, r) for p, s, r in zip(preds, srcs, refs) if numerals(s)]
    if withnum:
        n = len(withnum)
        deva = lambda t: any(c in DEVA for c in t)
        out |= {
            "nsem": round(sum(numerals(p) == numerals(s) for p, s, _ in withnum) / n * 100, 1),
            "nsem_vs_ref": round(sum(numerals(p) == numerals(r) for p, _, r in withnum) / n * 100, 1),
            "ref_preserves": round(sum(numerals(s) == numerals(r) for _, s, r in withnum) / n * 100, 1),
            "digit_script_match": round(sum(deva(p) == deva(r) for p, _, r in withnum) / n * 100, 1),
            "n_numeral": n,
        }

    out["exact"] = round(sum(norm(p) == norm(r) for p, r in zip(preds, refs)) / len(preds) * 100, 1)
    hard = [(p, s) for p, s, r in zip(preds, srcs, refs) if norm(s) != norm(r)]
    if hard:
        out["spurious_copy"] = round(sum(norm(p) == norm(s) for p, s in hard) / len(hard) * 100, 1)
        out["n_hard"] = len(hard)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--sets", nargs="+",
                    default=["GENERAL", "NUM_NAT", "SHORT_NAT", "SHORT_HARD"])
    ap.add_argument("--data-dir", type=Path, default=Path("data/bhili"))
    ap.add_argument("--max-items", type=int, default=400)
    args = ap.parse_args()

    from bodhan_genai.mt import IndicMTEngine
    from huggingface_hub import HfApi

    register_bhili()
    api = HfApi()
    # upload_file will not create a missing repo, it just fails and retries.
    api.create_repo(RESULTS_REPO, repo_type="model", exist_ok=True)
    out = Path("results")
    out.mkdir(exist_ok=True)

    metrics: dict[str, dict] = {}
    engine = IndicMTEngine("bodhan-ai/indic-translate", backend="hf",
                           **({"adapter_dir": args.adapter} if args.adapter else {}))
    with engine:
        for name in args.sets:
            path = args.data_dir / f"eval_{name}.jsonl"
            if not path.exists():
                print(f"  skip {name}")
                continue
            rows = [json.loads(l) for l in path.open(encoding="utf-8")][: args.max_items]
            mar = [r["mar"] for r in rows]
            bhb = [r["bhb"] for r in rows]

            for direction, srcs, refs, tgt in (("mar-bhb", mar, bhb, "Bhili"),
                                               ("bhb-mar", bhb, mar, "Marathi")):
                key = f"{name}.{direction}"
                print(f"  [{args.tag}] {key} n={len(srcs)}", flush=True)
                preds = [(x.text or "") for x in engine.translate_batch(srcs, tgt_lang=tgt)]

                gen = out / f"{args.tag}.{key}.jsonl"
                with gen.open("w", encoding="utf-8") as fh:
                    for s, p, r in zip(srcs, preds, refs):
                        fh.write(json.dumps({"src": s, "pred": p, "ref": r},
                                            ensure_ascii=False) + "\n")
                push(api, gen, f"generations/{gen.name}")

                metrics[key] = score(preds, srcs, refs)
                metrics[f"{key}.COPY"] = score(srcs, srcs, refs)
                print(f"    {metrics[key]}", flush=True)

                path_m = out / f"{args.tag}.metrics.json"
                path_m.write_text(json.dumps(metrics, indent=2))
                push(api, path_m, f"metrics/{path_m.name}")

    print(f"\n  {args.tag}: {len(metrics)} rows", flush=True)


if __name__ == "__main__":
    main()
