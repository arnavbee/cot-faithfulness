#!/usr/bin/env python3
"""Two detectors, one set of traces: how much of the headline is the classifier?

The headline in summary.json is the hint-acknowledgement rate among flips, and it
is 100% (44/44) under every detector in cotf/detect.py. Those detectors are three
phrasings of one lexical question, so their agreement is not evidence that the
number is robust. This script scores the same traces with a detector built to
disagree -- an LLM judge asked about load-bearingness, not mention -- and reports
the result as a range with an agreement statistic attached.

Design, fixed before looking at any judge output:

  sample      All 44 flips (a census, since that is the headline's own
              denominator), plus a stratified random 156 non-flips, seed 0.
              n = 200. The positive base rate is therefore enriched on purpose;
              every rate below is reported on its own denominator, never pooled
              across the two strata as if 200 were a random sample of 465.
  detector 1  cotf.detect.detect_strict -- regex, LEXICAL MENTION.
  detector 2  cotf.judge.judge_dependence -- LLM judge, EPISTEMIC DEPENDENCE,
              label C. Blind to the control answer and to flip status.
  asymmetry   The two off-diagonal counts are reported separately and never
              summed into a single "disagreement rate", because b and c mean
              opposite things about the metric.
  test        McNemar on the paired labels (exact binomial, plus the corrected
              chi-square for comparison).

Nothing here regenerates a trace. Judge verdicts are cached to disk, so a second
run is free and cannot produce different numbers.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cotf.detect import all_detectors
from cotf.hints import HINTS
from cotf.judge import (JudgeCache, _cache_key, judge_dependence,
                        build_prompt as build_judge_prompt)
from cotf.providers import get_provider
from cotf.runner import control_answer, load_run
from cotf.stats import bootstrap_rate, wilson

N_SAMPLE = 200
SEED = 0
PRIMARY_JUDGE = "openai/gpt-oss-20b"


# ---------------------------------------------------------------------------
# cases and sampling
# ---------------------------------------------------------------------------

def build_rows(run: str):
    """Every usable hinted response, with the regex verdicts attached."""
    cfg, items, store = load_run(run)
    by_qid = {i.qid: i for i in items}
    rows = []
    for rec in store.all():
        if rec["condition"] == "control" or rec.get("error"):
            continue
        item = by_qid.get(rec["qid"])
        if item is None:
            continue
        ctrl = control_answer(store, rec["qid"])
        if ctrl is None or rec["answer"] is None:
            continue
        target = rec["target"]
        hint_text = HINTS[rec["condition"]].render(item, target)
        exclude = item.question + " " + " ".join(item.choices)
        v = all_detectors(rec["cot"], rec["condition"], target, hint_text, exclude)
        rows.append({
            "qid": rec["qid"], "subject": item.subject, "hint": rec["condition"],
            "repeat": rec["repeat"], "target": target, "control_answer": ctrl,
            "hinted_answer": rec["answer"],
            "flipped": rec["answer"] == target and ctrl != target,
            "regex_strict": v["strict"].mentioned,
            "regex_loose": v["loose"].mentioned,
            "regex_verbatim": v["verbatim"].mentioned,
            "regex_evidence": v["strict"].evidence,
            "cot": rec["cot"],
        })
    return cfg, rows


def sample(rows, n=N_SAMPLE, seed=SEED):
    """All flips, then a stratified random fill from the non-flips.

    Stratified by hint type so the non-flip half cannot drift into whichever
    hint happens to produce the longest CoTs.
    """
    flips = [r for r in rows if r["flipped"]]
    nonflips = [r for r in rows if not r["flipped"]]
    need = n - len(flips)
    if need <= 0:
        return sorted(flips, key=_order)[:n]

    by_hint = defaultdict(list)
    for r in nonflips:
        by_hint[r["hint"]].append(r)
    total = len(nonflips)
    rng = random.Random(seed)
    chosen = []
    # Largest-remainder allocation: proportional shares that still sum to `need`.
    quotas = {h: need * len(v) / total for h, v in by_hint.items()}
    base = {h: int(q) for h, q in quotas.items()}
    short = need - sum(base.values())
    for h, _ in sorted(quotas.items(), key=lambda kv: (-(kv[1] - int(kv[1])), kv[0]))[:short]:
        base[h] += 1
    for h in sorted(by_hint):
        pool = sorted(by_hint[h], key=_order)
        rng.shuffle(pool)
        chosen += pool[:base[h]]
    return sorted(flips + chosen, key=_order)


def _order(r):
    return (r["qid"], r["hint"], r["repeat"])


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def table(a, b):
    """2x2 counts for paired binary labels a (detector 1) and b (detector 2)."""
    both = sum(1 for x, y in zip(a, b) if x and y)
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    neither = sum(1 for x, y in zip(a, b) if not x and not y)
    return {"both": both, "only_1": only_a, "only_2": only_b, "neither": neither}


def kappa(a, b):
    """Cohen's kappa with its CI. statsmodels where available, by hand otherwise."""
    t = table(a, b)
    n = sum(t.values())
    if n == 0:
        return None
    po = (t["both"] + t["neither"]) / n
    p1 = (t["both"] + t["only_1"]) / n
    p2 = (t["both"] + t["only_2"]) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    out = {"po": po, "pe": pe,
           "kappa": (po - pe) / (1 - pe) if pe < 1 else float("nan")}
    try:
        from statsmodels.stats.inter_rater import cohens_kappa
        res = cohens_kappa([[t["both"], t["only_1"]], [t["only_2"], t["neither"]]])
        out["kappa"] = float(res.kappa)
        out["se"] = float(res.std_kappa)
        out["lo"], out["hi"] = float(res.kappa_low), float(res.kappa_upp)
    except Exception as exc:            # pragma: no cover - optional dependency
        out["note"] = "statsmodels unavailable ({e}); kappa CI omitted".format(e=exc)
    return out


def mcnemar_test(a, b):
    t = table(a, b)
    out = {"b_only_1": t["only_1"], "c_only_2": t["only_2"]}
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        tbl = [[t["both"], t["only_1"]], [t["only_2"], t["neither"]]]
        ex = mcnemar(tbl, exact=True)
        ch = mcnemar(tbl, exact=False, correction=True)
        out["exact_statistic"] = float(ex.statistic)
        out["exact_p"] = float(ex.pvalue)
        out["chi2_statistic"] = float(ch.statistic)
        out["chi2_p"] = float(ch.pvalue)
    except Exception as exc:            # pragma: no cover
        out["note"] = "statsmodels unavailable ({e})".format(e=exc)
    return out


def rate(flags, clusters):
    """Bootstrap-cluster CI, clustering on qid, to match the rest of the repo."""
    if not flags:
        return None
    return bootstrap_rate(list(flags), list(clusters)).to_dict()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run(args):
    cfg, rows = build_rows(args.run)
    picked = sample(rows, args.n, args.seed)
    print("usable hinted cases: {u}   sampled: {s}   flips in sample: {f}".format(
        u=len(rows), s=len(picked), f=sum(r["flipped"] for r in picked)),
        file=sys.stderr)

    judges = [args.judge] + list(args.extra_judge or [])
    provider = get_provider(args.provider)
    outdir = Path("runs") / args.run
    for model in judges:
        cache = JudgeCache(outdir / ("judge-%s.jsonl" % model.replace("/", "_")))
        todo = sum(1 for r in picked if cache.get(_cache_key(
            model, build_judge_prompt(r["cot"], r["hint"], r["target"]))) is None)
        print("judge {m}: {t} of {n} uncached".format(m=model, t=todo, n=len(picked)),
              file=sys.stderr)
        for i, r in enumerate(picked, 1):
            v = judge_dependence(r["cot"], r["hint"], r["target"], provider,
                                 model, cache, max_tokens=args.max_tokens)
            r.setdefault("judge", {})[model] = {
                "label": v.label, "depends": v.depends, "mentions": v.mentions,
                "error": v.error}
            if i % 25 == 0 or v.error or not v.label:
                print("  [{i}/{n}] {q} {h} -> {l} {e}".format(
                    i=i, n=len(picked), q=r["qid"], h=r["hint"],
                    l=v.label or "??", e=v.error or ""), file=sys.stderr, flush=True)

    report = analyse(cfg, rows, picked, judges)

    labels_path = Path("labels") / ("sensitivity-%s.jsonl" % args.run)
    with labels_path.open("w") as fh:
        for r in picked:
            rec = {k: v for k, v in r.items() if k != "cot"}
            rec["cot_chars"] = len(r["cot"])
            fh.write(json.dumps(rec) + "\n")
    rep_path = outdir / "detector-sensitivity.json"
    rep_path.write_text(json.dumps(report, indent=2))
    print(format_report(report))
    print("\nper-trace labels -> {l}\nreport -> {r}".format(l=labels_path, r=rep_path))
    return 0


def analyse(cfg, rows, picked, judges):
    primary = judges[0]
    rep = {
        "run": cfg.name,
        "model_under_test": cfg.model,
        "n_usable_hinted": len(rows),
        "n_sampled": len(picked),
        "n_flips_in_sample": sum(r["flipped"] for r in picked),
        "sample_design": (
            "all flips (census) + stratified random non-flips, seed %d; positive "
            "base rate enriched on purpose" % SEED),
        "detector_1": {"name": "regex/detect_strict", "encodes": "lexical mention"},
        "detector_2": {"name": "judge_dependence/" + primary,
                       "encodes": "epistemic dependence (label C)"},
        "judges": judges,
        "strata": {},
        "judge_label_mix": {},
        "by_hint": {},
        "judge_vs_judge": {},
        "unreadable_judge_verdicts": {},
    }

    for model in judges:
        bad = [r["qid"] for r in picked if not r["judge"][model]["label"]]
        rep["unreadable_judge_verdicts"][model] = {"n": len(bad), "qids": bad[:10]}
        rep["judge_label_mix"][model] = dict(
            Counter(r["judge"][model]["label"] or "??" for r in picked))

    strata = {
        "all_sampled": picked,
        "flips_only": [r for r in picked if r["flipped"]],
        "nonflips_only": [r for r in picked if not r["flipped"]],
    }
    for name, subset in strata.items():
        if not subset:
            continue
        a = [r["regex_strict"] for r in subset]
        b = [r["judge"][primary]["depends"] for r in subset]
        clusters = [r["qid"] for r in subset]
        t = table(a, b)
        rep["strata"][name] = {
            "n": len(subset),
            "detector_1_rate": rate(a, clusters),
            "detector_2_rate": rate(b, clusters),
            "range_pct_points": round(100 * (sum(a) - sum(b)) / len(subset), 1),
            "table": t,
            "asymmetry": {
                "regex_yes_judge_no": t["only_1"],
                "regex_no_judge_yes": t["only_2"],
                "ratio": (t["only_1"] / t["only_2"]) if t["only_2"] else None,
            },
            "kappa": kappa(a, b),
            "mcnemar": mcnemar_test(a, b),
            "raw_agreement": wilson(t["both"] + t["neither"], len(subset)).to_dict(),
        }

    for hint in sorted({r["hint"] for r in picked}):
        subset = [r for r in picked if r["hint"] == hint]
        a = [r["regex_strict"] for r in subset]
        b = [r["judge"][primary]["depends"] for r in subset]
        t = table(a, b)
        rep["by_hint"][hint] = {
            "n": len(subset), "n_flips": sum(r["flipped"] for r in subset),
            "detector_1_k": sum(a), "detector_2_k": sum(b),
            "table": t, "kappa": kappa(a, b),
        }

    if len(judges) > 1:
        for other in judges[1:]:
            a = [r["judge"][primary]["depends"] for r in picked]
            b = [r["judge"][other]["depends"] for r in picked]
            t = table(a, b)
            rep["judge_vs_judge"][primary + " vs " + other] = {
                "rate_1": sum(a) / len(a), "rate_2": sum(b) / len(b),
                "table": t, "kappa": kappa(a, b), "mcnemar": mcnemar_test(a, b),
                "label_agreement": sum(
                    1 for r in picked
                    if r["judge"][primary]["label"] == r["judge"][other]["label"]
                ) / len(picked),
            }
    return rep


def format_report(rep):
    L = []
    w = L.append
    w("")
    w("DETECTOR SENSITIVITY  run={r}  model under test={m}".format(
        r=rep["run"], m=rep["model_under_test"]))
    w("  detector 1: {n:34s} encodes {e}".format(
        n=rep["detector_1"]["name"], e=rep["detector_1"]["encodes"]))
    w("  detector 2: {n:34s} encodes {e}".format(
        n=rep["detector_2"]["name"], e=rep["detector_2"]["encodes"]))
    w("  sampled {s} of {u} usable hinted traces; {f} are flips".format(
        s=rep["n_sampled"], u=rep["n_usable_hinted"], f=rep["n_flips_in_sample"]))
    for model, mix in rep["judge_label_mix"].items():
        w("  judge label mix {m}: {x}".format(m=model, x=mix))
    w("")
    for name, s in rep["strata"].items():
        t, k, mc = s["table"], s["kappa"], s["mcnemar"]
        d1, d2 = s["detector_1_rate"], s["detector_2_rate"]
        w("[{n}]  n={N}".format(n=name, N=s["n"]))
        w("  lexical mention    {p:6.1%}  [{lo:.1%}, {hi:.1%}]  k={k}".format(
            p=d1["point"], lo=d1["lo"], hi=d1["hi"], k=t["both"] + t["only_1"]))
        w("  epistemic depend.  {p:6.1%}  [{lo:.1%}, {hi:.1%}]  k={k}".format(
            p=d2["point"], lo=d2["lo"], hi=d2["hi"], k=t["both"] + t["only_2"]))
        w("  spread             {s:+.1f} points of the same 200 traces".format(
            s=s["range_pct_points"]))
        w("  2x2                both={b} regexOnly={o1} judgeOnly={o2} neither={n}".format(
            b=t["both"], o1=t["only_1"], o2=t["only_2"], n=t["neither"]))
        ratio = s["asymmetry"]["ratio"]
        w("  asymmetry          regex-yes/judge-no={a}  regex-no/judge-yes={b}{r}".format(
            a=t["only_1"], b=t["only_2"],
            r="   ({:.0f}:1)".format(ratio) if ratio else ""))
        if k:
            ci = ("  [{lo:.2f}, {hi:.2f}]".format(lo=k["lo"], hi=k["hi"])
                  if "lo" in k else "")
            w("  Cohen's kappa      {v:.3f}{c}   (observed {po:.1%}, chance {pe:.1%})".format(
                v=k["kappa"], c=ci, po=k["po"], pe=k["pe"]))
        if "exact_p" in mc:
            w("  McNemar exact      b={b} c={c}  p={p:.3g}   (chi2 corrected p={q:.3g})".format(
                b=mc["b_only_1"], c=mc["c_only_2"], p=mc["exact_p"], q=mc["chi2_p"]))
        w("")
    w("by hint (detector 1 k / detector 2 k of n, kappa):")
    for h, s in rep["by_hint"].items():
        kv = s["kappa"]["kappa"] if s["kappa"] else float("nan")
        w("  {h:16s} n={n:3d} flips={f:2d}   mention={a:3d}  depends={b:3d}   kappa={k:6.3f}".format(
            h=h, n=s["n"], f=s["n_flips"], a=s["detector_1_k"],
            b=s["detector_2_k"], k=kv))
    if rep["judge_vs_judge"]:
        w("")
        w("judge vs judge (classifier instability within the same question):")
        for pair, s in rep["judge_vs_judge"].items():
            w("  {p}".format(p=pair))
            w("    depends rate {a:.1%} vs {b:.1%}   3-way label agreement {l:.1%}   kappa={k:.3f}".format(
                a=s["rate_1"], b=s["rate_2"], l=s["label_agreement"],
                k=s["kappa"]["kappa"]))
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="main-40x3")
    p.add_argument("--provider", default="groq")
    p.add_argument("--judge", default=PRIMARY_JUDGE,
                   help="primary judge model (pre-registered)")
    p.add_argument("--extra-judge", action="append",
                   help="additional judge model, reported as a robustness column")
    p.add_argument("--n", type=int, default=N_SAMPLE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--max-tokens", type=int, default=2048)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
