"""Multiple-choice items, fetched once and cached to disk.

MMLU and GPQA-diamond are pulled from the HuggingFace datasets-server REST API,
which needs no key and no `datasets` install. Everything is cached under data/
so a re-run costs nothing and the item set is frozen for the life of an
experiment.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
LETTERS = "ABCDEFGH"

# subject -> the datasets-server config name
MMLU_SUBJECTS = [
    "high_school_mathematics",
    "professional_medicine",
    "high_school_world_history",
    "formal_logic",
    "college_physics",
]


@dataclass
class Item:
    qid: str
    question: str
    choices: List[str]
    answer: str          # correct letter
    subject: str
    source: str

    def letters(self) -> List[str]:
        return list(LETTERS[:len(self.choices)])

    def rendered_choices(self) -> str:
        return "\n".join("({l}) {c}".format(l=l, c=c)
                         for l, c in zip(self.letters(), self.choices))

    def to_dict(self) -> Dict:
        return asdict(self)


def _fetch_rows(dataset: str, config: str, split: str, offset: int, length: int) -> List[Dict]:
    out: List[Dict] = []
    while len(out) < length:
        batch = min(100, length - len(out))
        r = requests.get(ROWS_URL, params={
            "dataset": dataset, "config": config, "split": split,
            "offset": offset + len(out), "length": batch}, timeout=60)
        r.raise_for_status()
        rows = r.json().get("rows", [])
        if not rows:
            break
        out.extend(row["row"] for row in rows)
    return out


def fetch_mmlu(subjects: Optional[List[str]] = None, per_subject: int = 40,
               split: str = "test", cache: bool = True) -> List[Item]:
    subjects = subjects or MMLU_SUBJECTS
    items: List[Item] = []
    for subject in subjects:
        path = DATA_DIR / "mmlu_{s}_{n}_{sp}.json".format(s=subject, n=per_subject, sp=split)
        if cache and path.exists():
            raw = json.loads(path.read_text())
        else:
            raw = _fetch_rows("cais/mmlu", subject, split, 0, per_subject)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(raw, indent=1))
        for i, row in enumerate(raw):
            choices = list(row["choices"])
            items.append(Item(
                qid="mmlu/{s}/{i}".format(s=subject, i=i),
                question=row["question"].strip(),
                choices=choices,
                answer=LETTERS[int(row["answer"])],
                subject=subject,
                source="cais/mmlu",
            ))
    return items


def fetch_gpqa(n: int = 50, cache: bool = True) -> List[Item]:
    """GPQA-diamond. Gated on HF, so this needs HF_TOKEN; falls back loudly."""
    path = DATA_DIR / "gpqa_diamond_{n}.json".format(n=n)
    if cache and path.exists():
        raw = json.loads(path.read_text())
    else:
        raw = _fetch_rows("Idavidrein/gpqa", "gpqa_diamond", "train", 0, n)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw, indent=1))
    items: List[Item] = []
    for i, row in enumerate(raw):
        correct = row["Correct Answer"].strip()
        wrong = [row["Incorrect Answer 1"], row["Incorrect Answer 2"],
                 row["Incorrect Answer 3"]]
        # Deterministic shuffle so the correct answer is not always (A).
        rng = random.Random(i)
        choices = [correct] + [w.strip() for w in wrong]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [choices[j] for j in order]
        items.append(Item(
            qid="gpqa/diamond/{i}".format(i=i),
            question=row["Question"].strip(),
            choices=shuffled,
            answer=LETTERS[shuffled.index(correct)],
            subject=row.get("Subdomain", "gpqa"),
            source="Idavidrein/gpqa",
        ))
    return items


def load(name: str, n: int, seed: int = 0) -> List[Item]:
    """Load and deterministically subsample a dataset."""
    if name == "mmlu":
        per = max(1, (n // len(MMLU_SUBJECTS)) + 5)
        pool = fetch_mmlu(per_subject=per)
    elif name == "gpqa":
        pool = fetch_gpqa(n=max(n, 50))
    else:
        raise ValueError("unknown dataset '{n}'".format(n=name))
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]
