#!/usr/bin/env python3
"""Fold the hand adjudication into the sensitivity result and print the range.

`detector_sensitivity.py` gives two machine opinions and the distance between
them. It cannot say which end of the range is closer to the truth. That needs a
human pass, and the honest place to spend a human pass is the disagreement cells
plus a sample of the agreements (to bound the error in the direction the
disagreement cells cannot see).

What was read by hand:
  - all 14 flips where regex said mention and the judge said not-load-bearing
  - 8 of the 30 flips where both said yes, to check the judge's C precision
  - 6 of the 63 non-flips where regex said mention and the judge said B
The 22 unread both-yes flips are accepted as C. That is an assumption, and it is
the one to attack first if this result matters: 8 of 8 sampled were correct, but
8 of 8 bounds the precision loosely (Wilson lower bound ~0.68).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cotf.stats import bootstrap_rate, wilson

JUDGE = "openai/gpt-oss-20b"
SPOTCHECK_AGREE_N, SPOTCHECK_AGREE_OK = 8, 8


def main():
    labels = [json.loads(l) for l in open("labels/sensitivity-main-40x3.jsonl")]
    adj = {(r["qid"], r["hint"], r["repeat"]): r
           for r in map(json.loads, open("labels/adjudication-flips.jsonl"))}

    flips = [r for r in labels if r["flipped"]]
    regex = [r["regex_strict"] for r in flips]
    judge = [r["judge"][JUDGE]["depends"] for r in flips]
    human = []
    for r in flips:
        k = (r["qid"], r["hint"], r["repeat"])
        if k in adj:
            human.append(adj[k]["human"] == "C")
        else:
            human.append(True)          # accepted judge C, see module docstring
    clusters = [r["qid"] for r in flips]

    def rate(flags, label):
        br = bootstrap_rate(list(flags), clusters)
        return "{l:24s} {p:6.1%}  [{lo:.1%}, {hi:.1%}]   k={k}/{n}".format(
            l=label, p=br.point, lo=br.lo, hi=br.hi, k=sum(flags), n=len(flags))

    n = len(flips)
    print("\nHEADLINE: hint-acknowledgement among flips, n={n} (census, not a sample)".format(n=n))
    print("  " + rate(regex, "regex / lexical mention"))
    print("  " + rate(judge, "LLM judge / dependence"))
    print("  " + rate(human, "human adjudication"))
    print("  spread regex-judge   {d:.1f} points".format(
        d=100 * (sum(regex) - sum(judge)) / n))
    print("  spread regex-human   {d:.1f} points".format(
        d=100 * (sum(regex) - sum(human)) / n))

    print("\nWHICH DETECTOR WAS WRONG (against the human pass, flips only)")
    tp = sum(1 for j, h in zip(judge, human) if j and h)
    fp = sum(1 for j, h in zip(judge, human) if j and not h)
    fn = sum(1 for j, h in zip(judge, human) if not j and h)
    tn = sum(1 for j, h in zip(judge, human) if not j and not h)
    print("  judge vs human   tp={tp} fp={fp} fn={fn} tn={tn}".format(
        tp=tp, fp=fp, fn=fn, tn=tn))
    print("    judge recall    {r:.1%}  [{lo:.1%}, {hi:.1%}]   <- the whole gap is here".format(
        **{"r": tp / (tp + fn), **{k: v for k, v in
           zip(("lo", "hi"), (wilson(tp, tp + fn).lo, wilson(tp, tp + fn).hi))}}))
    print("    judge precision {p:.1%}   (fp={fp}; 8/8 of the sampled agreements held)".format(
        p=tp / (tp + fp) if tp + fp else float("nan"), fp=fp))
    rtp = sum(1 for g, h in zip(regex, human) if g and h)
    rfp = sum(1 for g, h in zip(regex, human) if g and not h)
    print("  regex vs human   tp={tp} fp={fp} fn=0 tn=0".format(tp=rtp, fp=rfp))
    print("    regex precision {p:.1%}   recall 100.0%".format(p=rtp / (rtp + rfp)))

    print("\nKAPPA, and why one of these is uninformative")
    for a, b, name in ((regex, judge, "regex vs judge"),
                       (judge, human, "judge vs human"),
                       (regex, human, "regex vs human")):
        both = sum(1 for x, y in zip(a, b) if x and y)
        neither = sum(1 for x, y in zip(a, b) if not x and not y)
        po = (both + neither) / n
        p1, p2 = sum(a) / n, sum(b) / n
        pe = p1 * p2 + (1 - p1) * (1 - p2)
        k = (po - pe) / (1 - pe) if pe < 1 else float("nan")
        deg = " DEGENERATE: one rater has no variance, kappa is 0 by construction" \
            if sum(a) in (0, n) or sum(b) in (0, n) else ""
        print("  {nm:16s} observed {po:6.1%}  chance {pe:6.1%}  kappa {k:6.3f}{d}".format(
            nm=name, po=po, pe=pe, k=k, d=deg))
    print("\n  pooled over all 200 sampled traces (enriched base rate, regex does vary):")
    rep = json.load(open("runs/main-40x3/detector-sensitivity.json"))
    s = rep["strata"]["all_sampled"]
    print("    regex vs judge   kappa {k:.3f}  [{lo:.2f}, {hi:.2f}]   "
          "McNemar b={b} c={c} p={p:.3g}".format(
              k=s["kappa"]["kappa"], lo=s["kappa"]["lo"], hi=s["kappa"]["hi"],
              b=s["mcnemar"]["b_only_1"], c=s["mcnemar"]["c_only_2"],
              p=s["mcnemar"]["exact_p"]))

    out = {
        "n_flips": n,
        "regex_lexical_mention": {"k": sum(regex), "rate": sum(regex) / n},
        "judge_dependence": {"k": sum(judge), "rate": sum(judge) / n},
        "human_adjudicated": {"k": sum(human), "rate": sum(human) / n},
        "reported_range": [min(sum(judge), sum(human), sum(regex)) / n,
                           max(sum(judge), sum(human), sum(regex)) / n],
        "judge_vs_human": {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                           "recall": tp / (tp + fn),
                           "precision": tp / (tp + fp) if tp + fp else None},
        "regex_vs_human": {"tp": rtp, "fp": rfp,
                           "precision": rtp / (rtp + rfp), "recall": 1.0},
        "pooled_200_kappa": s["kappa"]["kappa"],
        "assumption": ("22 of the 30 both-yes flips accepted as C without "
                       "individual re-reading; 8 of 8 sampled held"),
    }
    Path("runs/main-40x3/adjudicated-range.json").write_text(json.dumps(out, indent=2))
    print("\n-> runs/main-40x3/adjudicated-range.json")


if __name__ == "__main__":
    main()
