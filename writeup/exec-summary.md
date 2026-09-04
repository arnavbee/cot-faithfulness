# Executive summary — draft for Arnav to rewrite in his own voice before pasting

Neel's rule, verbatim: "Please do not just submit raw LLM output for the application form or executive summary. Write these yourself, in your own voice." So this file is the *content*; the sentences are yours to write.

## The question
When a hint pushes gpt-oss-120b to a wrong multiple-choice answer, does its chain of thought say so?

## What I found (four numbers, one sentence each)
- Grader hint flipped 31.9% of answers [19.3, 45.0]; expert 5.3%; sycophancy and metadata 0%. Overall 9.5% on 465 hinted responses vs a 0.83% no-hint baseline.
- All 44 flips acknowledged the hint in the reasoning field; I hand-read every one blind and agreed with the detector on 43. Unfaithful rate 0%.
- Same grader marked "internal, not for the model": flips fall to 2.6%. Legitimacy, not visibility, drives compliance.
- Same 44 flips, visible answer instead of reasoning: only 2 mention the hint, 34 are a bare "Answer: (X)". One reasoning chain says "we must not reveal that we are cheating."

## Why it matters
The faithfulness question has two channels. On this model the private reasoning is fully faithful and the user-facing answer almost never is, and the model states the split as a plan. Which channel a monitor is allowed to read decides whether it catches anything.

## Biggest limitation
One model, 40 items, and the grader hint is ordinary user-turn text that the model reads as a legitimate spec. Two hint types produced zero flips and are untested, not disproven.

## What I would do with more time
A legitimacy gradient for the grader framing, and the same output-channel check on a model whose visible CoT *is* its reasoning.

## How I checked myself
Read all 44 flips by hand blind (98% detector agreement), re-derived the headline from raw rows (exact match), read all 15 parse failures (13 are token-cap, 3 items), reconciled the two denominators. Section 5 of the write-up.
