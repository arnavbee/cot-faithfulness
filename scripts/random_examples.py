#!/usr/bin/env python3
"""Randomly selected qualitative examples for the write-up (Neel: "randomly selected, not cherry-picked").

Seeded, so the selection is reproducible and provably not hand-picked:
  3 flips drawn from labels/<run>.jsonl, 2 non-flip hinted responses drawn from runs/<run>/responses.jsonl.
Prints markdown to stdout.  .venv/bin/python scripts/random_examples.py [run] [seed]
"""
import json, random, sys
from pathlib import Path
RUN = sys.argv[1] if len(sys.argv) > 1 else "main-40x3"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ROOT = Path(__file__).resolve().parents[1]
labels = [json.loads(l) for l in (ROOT / "labels" / f"{RUN}.jsonl").open()]
resp = {}
for l in (ROOT / "runs" / RUN / "responses.jsonl").open():
    r = json.loads(l)
    if not r.get("error"):
        resp[(r["qid"], r["condition"], r["repeat"])] = r
flip_keys = {(r["qid"], r["hint"], r["repeat"]) for r in labels}
hinted_nonflip = [k for k in resp if k[1] not in ("", "control") and k not in flip_keys and resp[k].get("answer")]
rng = random.Random(SEED)
flips = rng.sample(labels, 3)
nonflips = rng.sample(sorted(hinted_nonflip), 2)

def clip(s, n):
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:n].rstrip() + " […]"

def hint_line(prompt):
    return prompt.split("\n\n")[0].replace("\n", " ")

print(f"Selected with `scripts/random_examples.py {RUN} {SEED}`: 3 of the 44 flips and 2 of the 421 hinted non-flips, seeded random draw.\n")
for i, r in enumerate(flips, 1):
    rec = resp[(r["qid"], r["hint"], r["repeat"])]
    print(f"**Flip {i}. {r['qid']}, hint `{r['hint']}`, repeat {r['repeat']}.** Control answer {r['control_answer']}, hinted answer {r['hinted_answer']}, target {r['target']}. Hand label: {'Y' if r['human'] else 'N'}.")
    print(f"- Hint as shown: `{clip(hint_line(rec['prompt']), 160)}`")
    print(f"- Reasoning field (first 500 chars): {clip(r['cot'], 500)}")
    print(f"- Visible answer (last 200 chars): {clip(rec['text'][-200:], 200)}\n")
for i, k in enumerate(nonflips, 1):
    rec = resp[k]
    print(f"**Non-flip {i}. {k[0]}, hint `{k[1]}`, repeat {k[2]}.** Answer {rec['answer']}, target {rec['target']}.")
    print(f"- Hint as shown: `{clip(hint_line(rec['prompt']), 160)}`")
    print(f"- Reasoning field (first 400 chars): {clip(rec.get('cot',''), 400)}")
    print(f"- Visible answer (last 160 chars): {clip(rec['text'][-160:], 160)}\n")
