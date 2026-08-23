"""Hand-labelling support.

Neel's doc calls sanity-checking the single most important thing, and names
"I read 30 transcripts and confirmed the positives were real" as strong
evidence. This module makes that cheap and makes the result a number.

  cotf label <run> --n 30      -> writes labels/<run>.jsonl with blank verdicts
  (you fill in "human": true/false by hand)
  cotf calibrate <run>         -> agreement between each detector and your labels
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

from .stats import score_classifier, wilson

LABEL_DIR = Path(__file__).resolve().parent.parent / "labels"


def sample_for_labelling(cases: List, n: int, run_name: str, seed: int = 0) -> Path:
    """Sample flips stratified by hint type, blank verdict field, one per line."""
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    path = LABEL_DIR / "{r}.jsonl".format(r=run_name)
    flips = [c for c in cases if c.flipped]
    by_hint: Dict[str, List] = {}
    for c in flips:
        by_hint.setdefault(c.hint, []).append(c)

    rng = random.Random(seed)
    per = max(1, n // max(1, len(by_hint)))
    chosen = []
    for hint, group in sorted(by_hint.items()):
        rng.shuffle(group)
        chosen.extend(group[:per])
    rng.shuffle(chosen)
    chosen = chosen[:n]

    existing = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                existing[(rec["qid"], rec["hint"], rec["repeat"])] = rec.get("human")

    with path.open("w") as fh:
        for c in chosen:
            key = (c.qid, c.hint, c.repeat)
            fh.write(json.dumps({
                "qid": c.qid, "hint": c.hint, "repeat": c.repeat,
                "target": c.target, "control_answer": c.control_answer,
                "hinted_answer": c.hinted_answer,
                "detectors": c.mentions,
                "human": existing.get(key),   # <- you fill this in: true or false
                "cot": c.cot,
            }) + "\n")
    return path


def calibrate(run_name: str) -> str:
    path = LABEL_DIR / "{r}.jsonl".format(r=run_name)
    if not path.exists():
        return "NO_LABELS: run `cotf label {r}` first".format(r=run_name)
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    done = [r for r in recs if isinstance(r.get("human"), bool)]
    if not done:
        return ("NO_LABELS_FILLED: {n} rows waiting in {p}. Set \"human\": true or "
                "false on each.".format(n=len(recs), p=path))

    lines = ["", "DETECTOR CALIBRATION against {n} hand labels".format(n=len(done)),
             "  {d:<12} {a:>9} {p:>7} {r:>7}".format(
                 d="detector", a="agreement", p="prec", r="recall")]
    truth = [r["human"] for r in done]
    for name in ["verbatim", "strict", "loose"]:
        pred = [bool(r["detectors"].get(name)) for r in done]
        agree = sum(1 for p, t in zip(pred, truth) if p == t)
        s = score_classifier(pred, truth)
        rate = wilson(agree, len(done))
        lines.append("  {d:<12} {a:>9} {p:>7} {r:>7}".format(
            d=name, a="{v:.0%}".format(v=rate.point),
            p="{v:.2f}".format(v=s.precision) if s.precision == s.precision else "n/a",
            r="{v:.2f}".format(v=s.recall) if s.recall == s.recall else "n/a"))
    lines.append("")
    lines.append("  Human labels are ground truth here. If your primary detector")
    lines.append("  disagrees with you on more than ~10% of cases, fix the detector")
    lines.append("  before reporting the headline number.")
    lines.append("")
    return "\n".join(lines)
