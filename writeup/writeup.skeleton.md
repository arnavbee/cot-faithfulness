# [TITLE — one line, Arnav's words]

MATS 12.0 — Neel Nanda stream. Task write-up.
Run: `main-40x3` (groq / openai/gpt-oss-120b, MMLU, 40 items x 5 conditions x 3 repeats = 600 calls, temperature 0, seed 0).
All numbers below are pasted verbatim from `runs/main-40x3/summary.json` and `cost.json`, generated 2026-09-01 22:25 IST.

> SKELETON ONLY. Every prose block below is a slot, not a draft. Nothing here is written for you.

---

## 1. Assertion

[One falsifiable sentence, your words. The pre-clock checklist says this is yours, not mine.]

## 2. What I did

### 2.1 Paradigm
[Hint injection, Turpin/Chen lineage. Two sentences.]

### 2.2 Conditions
| Condition | What the hint is |
|---|---|
| control | no hint |
| sycophancy | user states their own belief before the model answers |
| metadata | XML metadata block containing an answer field |
| authority | an unnamed expert is said to endorse an option |
| reward_hack | a visible grader function that returns True for one option |

### 2.3 The two definitions the experiment lives or dies on
- `flipped()` — [your definition]
- `mentions_hint()` — [your definition; three detectors shipped: verbatim / strict / loose. Primary = strict.]

### 2.4 Item set and ugly cases
[Which 40 items, and which of them you deliberately made hard. Yours.]

---

## 3. Results

![flip rate by hint](../figures/main-40x3_by_hint.png)

### 3.1 Headline — flip rate by hint (strict detector, bootstrap-cluster CI)

| Hint | n responses | flips | flip rate | 95% CI |
|---|---|---|---|---|
| reward_hack | 119 | 38 | **31.9%** | [19.3, 45.0] |
| authority | 113 | 6 | **5.3%** | [0.0, 12.4] |
| sycophancy | 116 | 0 | **0.0%** | [0.0, 0.0] |
| metadata | 117 | 0 | **0.0%** | [0.0, 0.0] |
| **overall** | 465 | 44 | **9.5%** | [5.6, 13.6] |

### 3.2 The negative result — acknowledgement

| Metric | Value | CI | n |
|---|---|---|---|
| acknowledgement rate among flips | **100%** | [100, 100] | 44 |
| **unfaithful rate** (flip, no mention) | **0.0%** | [0.0, 0.0] | 44 |

Detector spread — verbatim, strict and loose all return 1.0 on all 44 flips. The result does not
depend on which grader definition is used.

[Interpretation. Yours. This is the load-bearing paragraph of the whole write-up.]

### 3.3 Control and run hygiene

| Check | Value |
|---|---|
| completeness | 600/600 usable, 0 missing, 40/40 items complete |
| API error rate | 0.0% [0.0, 0.64] |
| parse failure rate | 2.5% [1.5, 4.1] (15/600) |
| spontaneous switch rate (control, no hint) | 0.83% [0.0, 2.5] (1/120) |

[Why the control matters: the 0.83% baseline is what the flip rates have to beat.]

### 3.4 Caveats the harness raised itself
- sycophancy: 0 flips, acknowledgement interval too wide to support any claim.
- metadata: 0 flips, same.
- authority: only 6 flips, same.
- So **only reward_hack carries a claim about acknowledgement.** [Say this out loud rather than burying it.]

---

## 4. The monitor

Task: read only the CoT, predict "this answer was hint-driven". Compared against baselines, per Neel's
"compare to baselines" rule.

![monitors](../figures/main-40x3_monitors.png)

| Monitor | Regime | Precision | Recall | F1 | 95% CI | Degenerate |
|---|---|---|---|---|---|---|
| target_letter_emphasis | oracle | 0.600 | 0.875 | **0.712** | [0.561, 0.831] | no |
| oracle_string_match | oracle | 0.643 | 0.750 | **0.692** | [0.524, 0.828] | no |
| hedging_markers | blind | 0.230 | 0.583 | **0.329** | [0.194, 0.455] | no |
| always_positive | trivial | 0.082 | 1.000 | 0.152 | [0.098, 0.209] | **yes** |
| certainty_markers | blind | 0.082 | 1.000 | 0.152 | [0.098, 0.209] | **yes** |
| cot_length | blind | 0.072 | 0.833 | 0.132 | [0.082, 0.185] | no |

Base rate 8.2%.

[Interpretation. The honest read is that no blind monitor beats the oracle string-match, and two of the
blind ones are degenerate. Say what that means. Yours.]

📌 **Say this explicitly or it reads as a discrepancy.** These monitors are scored on a
**held-out test split, split by item** (`monitors.split_by_item`, seed 0): 292 responses, 24 positives,
base rate 8.2%. The headline flip numbers in section 3 are over the **full 465 hinted responses, 44
flips**, 9.5%. Same run, different denominator. State the split in the text; Neel checks exactly this
kind of number-vs-claim mismatch.

---

## 5. Sanity checking

Neel: "the most important piece of advice in this doc." This section is evidence of research skill, not filler.

- [ ] Read N raw transcripts from `runs/main-40x3/responses.jsonl` yourself. Record N.
- [ ] Re-derive at least two load-bearing numbers by hand, independently of the harness.
- [ ] Be suspicious of the 0% unfaithful rate. A perfect 100% acknowledgement is exactly the shape of a
      grader bug. Read the flips and confirm the mentions are real, not artefacts of a loose regex.
- [ ] Read the 15 parse failures. Are they random, or concentrated in one condition?
- [ ] Reconcile the 292-vs-465 monitor discrepancy above.

[Write what you actually did and what you found. Numbers, not adjectives.]

---

## 6. Cost and compute

| | |
|---|---|
| calls | 600 (0 errored) |
| tokens | 382,468 total (150,750 prompt / 231,718 completion) |
| spend | **$0.16** |
| reasoning tokens | 201,800 = 87.1% of completion, $0.121 of the spend |
| cost per flip | $0.0037 |
| mean service latency | 1.15s (p95 2.45s) |
| throttle share of wall clock | 99.8% — free-tier rate limits, not model time |
| failed calls not billed | 1,676 (836 retries exhausted, 828 daily rate limit, 12 rate limit) |

[Optional: one line on what the 99.8% throttle share means for anyone reproducing this on a free tier.]

---

## 7. Limitations

- [One model only. Say which and why.]
- [40 items, MMLU only.]
- [0 flips on two conditions means two of four hint types are untested, not disproven.]
- [Yours.]

## 8. What I would do next

[Yours.]

---

## Appendix — how to reproduce
```
cotf run main-40x3
cotf score main-40x3
cotf cost main-40x3
cotf monitors main-40x3
```

## Appendix — LLM use
[Neel asks for this and rewards agentic use ~3x. State plainly: harness code, plots and tooling were
agent-written; the design, controls, interpretation, sanity-checking and this prose are yours.]
