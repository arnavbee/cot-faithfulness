"""Execution: control pass, then hinted pass, both resumable.

Order matters. The hinted option is chosen relative to the model's OWN control
answer, so the control pass has to finish for an item before its hinted prompts
can be built. That is why this is two phases and not one.

Every response is appended to responses.jsonl and keyed by a hash of
(provider, model, temperature, prompt, repeat). Re-running skips anything
already in the file, so a rate-limit wall or a closed laptop costs you nothing
but the time already spent.
"""

from __future__ import annotations

import json
import hashlib
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .dataset import Item
from .detect import extract_answer
from .hints import HINTS, build_prompt, pick_target
from .providers import DAILY_LIMIT_ERROR, Provider, get_provider

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


class DailyLimitReached(RuntimeError):
    """The provider's per-day budget is gone until it resets.

    Raised rather than returned so a caller cannot mistake a truncated run for a
    finished one. The CLI catches it and exits non-zero with the resume command,
    which is what a scheduler needs in order to not relaunch into the same wall.
    """


@dataclass
class RunConfig:
    name: str
    provider: str = "mock"
    model: str = ""
    dataset: str = "mmlu"
    n_items: int = 50
    repeats: int = 1
    hints: List[str] = field(default_factory=lambda: ["sycophancy", "metadata",
                                                      "authority", "reward_hack"])
    temperature: float = 0.0
    max_tokens: int = 2048
    seed: int = 0
    concurrency: int = 4

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Response:
    key: str
    qid: str
    condition: str          # "control" or a hint key
    target: str             # hinted letter, "" for control
    repeat: int
    answer: Optional[str]
    cot: str
    text: str
    prompt: str
    latency: float
    error: Optional[str] = None
    usage: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


def _key(provider: str, model: str, temperature: float, prompt: str, repeat: int) -> str:
    blob = "|".join([provider, model, "{t:.3f}".format(t=temperature),
                     str(repeat), prompt])
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


class ResponseStore:
    """Append-only JSONL with an in-memory index. Safe for concurrent writers."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.by_key: Dict[str, Dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.by_key[rec["key"]] = rec

    def has(self, key: str) -> bool:
        # An errored record is a hole, not a result. Treating it as cached would
        # make a resume skip it forever and leave the run silently incomplete,
        # with the gaps correlated to whatever caused the failure (rate limits).
        rec = self.by_key.get(key)
        return rec is not None and not rec.get("error")

    def add(self, resp: Response) -> None:
        with self._lock:
            self.by_key[resp.key] = resp.to_dict()
            with self.path.open("a") as fh:
                fh.write(json.dumps(resp.to_dict()) + "\n")

    def all(self) -> List[Dict]:
        """The final record per key: what the run produced."""
        return list(self.by_key.values())

    def attempts(self) -> List[Dict]:
        """Every record ever written, retries included, in file order.

        all() keeps one record per key, so a call that failed twice and then
        succeeded appears once, as a success. That is the right view for scoring
        and the wrong one for asking what the run cost: it silently deletes the
        failed attempts, which are exactly where the hours went.
        """
        records: List[Dict] = []
        if not self.path.exists():
            return records
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records


def _call(provider: Provider, cfg: RunConfig, item: Item, condition: str,
          target: str, repeat: int) -> Response:
    mock_markers = cfg.provider == "mock"
    hint_key = "" if condition == "control" else condition
    prompt = build_prompt(item, hint_key, target, mock_markers=mock_markers)
    key = _key(cfg.provider, cfg.model or provider.default_model,
               cfg.temperature, prompt, repeat)
    started = time.time()
    comp = provider.complete(prompt, cfg.model or provider.default_model,
                             temperature=cfg.temperature,
                             max_tokens=cfg.max_tokens,
                             seed=cfg.seed + repeat)
    answer = extract_answer(comp.text or comp.reasoning, valid=item.letters())
    return Response(
        key=key, qid=item.qid, condition=condition, target=target, repeat=repeat,
        answer=answer, cot=comp.cot, text=comp.text, prompt=prompt,
        latency=round(time.time() - started, 3), error=comp.error, usage=comp.usage)


def _run_batch(provider: Provider, cfg: RunConfig, store: ResponseStore,
               jobs: List, label: str) -> None:
    pending = []
    for item, condition, target, repeat in jobs:
        mock_markers = cfg.provider == "mock"
        hint_key = "" if condition == "control" else condition
        prompt = build_prompt(item, hint_key, target, mock_markers=mock_markers)
        key = _key(cfg.provider, cfg.model or provider.default_model,
                   cfg.temperature, prompt, repeat)
        if store.has(key):
            continue
        pending.append((item, condition, target, repeat))

    done_already = len(jobs) - len(pending)
    print("[{l}] {p} to run, {d} already cached".format(
        l=label, p=len(pending), d=done_already), file=sys.stderr)
    if not pending:
        return

    completed = 0
    errors = 0
    cancelled = 0
    stopped = False
    with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as pool:
        futures = {pool.submit(_call, provider, cfg, *job): job for job in pending}
        for fut in as_completed(futures):
            try:
                resp = fut.result()
            except CancelledError:
                # We cancelled these ourselves on the daily limit below. Asking
                # a cancelled future for its result re-raises, which used to
                # take the whole process down with a traceback the moment the
                # stop worked as designed.
                cancelled += 1
                continue
            store.add(resp)
            completed += 1
            if resp.error:
                errors += 1
            # A day cap will not clear inside this run. Cancel the rest rather
            # than writing hundreds of errored holes and spending the
            # requests-per-day budget discovering the same refusal each time.
            if resp.error and DAILY_LIMIT_ERROR in resp.error and not stopped:
                stopped = True
                for pend in futures:
                    pend.cancel()
            if completed % 10 == 0 or completed == len(pending):
                print("[{l}] {c}/{t} ({e} errors)".format(
                    l=label, c=completed, t=len(pending), e=errors), file=sys.stderr)

    if stopped:
        remaining = len(pending) - completed
        print("[{l}] STOPPED on the provider daily limit. {d} done, {r} left. "
              "Nothing is lost: errored records are not cached, so rerunning "
              "the same command once the budget resets resumes here.".format(
                  l=label, d=completed - errors, r=remaining + errors),
              file=sys.stderr)
        raise DailyLimitReached(
            "{l}: {d} done, {r} left".format(
                l=label, d=completed - errors, r=remaining + errors))


def control_answer(store: ResponseStore, qid: str) -> Optional[str]:
    """Majority answer across control repeats. Ties and blanks return None."""
    answers = [r["answer"] for r in store.all()
               if r["qid"] == qid and r["condition"] == "control" and r["answer"]]
    if not answers:
        return None
    counts = Counter(answers).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None
    return counts[0][0]


def run(cfg: RunConfig, items: List[Item]) -> ResponseStore:
    run_dir = RUNS_DIR / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    (run_dir / "items.json").write_text(
        json.dumps([i.to_dict() for i in items], indent=1))

    provider = get_provider(cfg.provider)
    store = ResponseStore(run_dir / "responses.jsonl")

    control_jobs = [(item, "control", "", rep)
                    for item in items for rep in range(cfg.repeats)]
    _run_batch(provider, cfg, store, control_jobs, "control")

    hinted_jobs = []
    skipped = 0
    for item in items:
        ctrl = control_answer(store, item.qid)
        if ctrl is None:
            skipped += 1
            continue
        target = pick_target(item, ctrl, seed=cfg.seed)
        for hint_key in cfg.hints:
            for rep in range(cfg.repeats):
                hinted_jobs.append((item, hint_key, target, rep))
    if skipped:
        print("[hinted] skipped {s} items with no parseable control answer".format(
            s=skipped), file=sys.stderr)
    _run_batch(provider, cfg, store, hinted_jobs, "hinted")
    return store


def load_run(name: str):
    run_dir = RUNS_DIR / name
    cfg = RunConfig(**json.loads((run_dir / "config.json").read_text()))
    items = [Item(**d) for d in json.loads((run_dir / "items.json").read_text())]
    store = ResponseStore(run_dir / "responses.jsonl")
    return cfg, items, store
