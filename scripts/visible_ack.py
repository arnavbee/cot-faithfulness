#!/usr/bin/env python3
"""Output-channel check: does the VISIBLE answer (text field) mention the hint, for every hand-labelled flip?

The headline acknowledgement rate is computed on the raw reasoning field (cot). This re-runs the same
detectors on the user-facing completion (text) for the same 44 flips. Also counts "bare" answers: visible
outputs whose only content is the final `Answer: (X)` line, i.e. compliance with no stated reasoning at all.

  .venv/bin/python scripts/visible_ack.py [run]      default run = main-40x3
"""
import json, re, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cotf.detect import detect_strict, detect_loose

RUN = sys.argv[1] if len(sys.argv) > 1 else "main-40x3"
ROOT = Path(__file__).resolve().parents[1]
rows = [json.loads(l) for l in (ROOT / "labels" / f"{RUN}.jsonl").open()]
resp = {}
for l in (ROOT / "runs" / RUN / "responses.jsonl").open():
    r = json.loads(l)
    if not r.get("error"):
        resp[(r["qid"], r["condition"], r["repeat"])] = r

def mentioned(v):
    return bool(getattr(v, "mentioned", v))

def is_bare(text):
    body = re.sub(r"Answer:\s*\(?[A-D]\)?\s*$", "", text.strip(), flags=re.I).strip()
    return len(body) == 0

cot_hits = Counter(); txt_hits = Counter(); bare = Counter(); n = Counter()
for r in rows:
    rec = resp[(r["qid"], r["hint"], r["repeat"])]
    h = r["hint"]; n[h] += 1
    cot_hits[h] += mentioned(detect_strict(r["cot"], h, r["target"]))
    txt_hits[h] += mentioned(detect_strict(rec["text"], h, r["target"]))
    bare[h] += is_bare(rec["text"])

print(f"OUTPUT-CHANNEL CHECK, run {RUN}, {sum(n.values())} hand-labelled flips (strict detector)")
print(f"  {'hint':14} {'n':>3} {'cot mentions':>13} {'text mentions':>14} {'bare answer':>12}")
for h in sorted(n):
    print(f"  {h:14} {n[h]:3d} {cot_hits[h]:13d} {txt_hits[h]:14d} {bare[h]:12d}")
N = sum(n.values())
print(f"  {'TOTAL':14} {N:3d} {sum(cot_hits.values()):13d} {sum(txt_hits.values()):14d} {sum(bare.values()):12d}")
print(f"\n  reasoning field acknowledges the hint: {sum(cot_hits.values())}/{N}")
print(f"  visible answer acknowledges the hint:  {sum(txt_hits.values())}/{N}")
print(f"  visible answer is a bare 'Answer: (X)' with no reasoning: {sum(bare.values())}/{N}")
