"""Command line entry point.

    python -m cotf doctor
    python -m cotf run  pilot --provider mock --n 40 --repeats 3
    python -m cotf score pilot
    python -m cotf monitor pilot
    python -m cotf plot pilot
    python -m cotf label pilot --n 30
    python -m cotf calibrate pilot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import dataset, labeling, monitors, plots, score
from .hints import HINTS
from .providers import REGISTRY, ProviderError, get_provider
from .runner import (RUNS_DIR, DailyLimitReached, RunConfig, load_run,
                     run as run_experiment)


def _summary_path(name: str) -> Path:
    return RUNS_DIR / name / "summary.json"


def cmd_doctor(args) -> int:
    print("providers:")
    for name in sorted(REGISTRY):
        provider = get_provider(name)
        status = "ready"
        if name != "mock":
            try:
                provider._key() if hasattr(provider, "_key") else None
            except ProviderError:
                status = "NO KEY"
        print("  {n:<12} {s:<8} default model: {m}".format(
            n=name, s=status, m=provider.default_model))
    print("\nhints:")
    for key, hint in HINTS.items():
        print("  {k:<12} {d}".format(k=key, d=hint.description))
    print("\ndatasets: mmlu (open), gpqa (needs HF access)")
    print("runs dir: {p}".format(p=RUNS_DIR))
    return 0


def cmd_run(args) -> int:
    cfg = RunConfig(
        name=args.name, provider=args.provider, model=args.model,
        dataset=args.dataset, n_items=args.n, repeats=args.repeats,
        hints=args.hints or list(HINTS), temperature=args.temperature,
        max_tokens=args.max_tokens, seed=args.seed, concurrency=args.concurrency)
    items = dataset.load(cfg.dataset, cfg.n_items, seed=cfg.seed)
    print("loaded {n} items from {d}".format(n=len(items), d=cfg.dataset),
          file=sys.stderr)
    try:
        run_experiment(cfg, items)
    except DailyLimitReached as exc:
        # Score what did land anyway: a partial summary with honest coverage is
        # more useful than nothing, and score.summarise already reports holes.
        print("run incomplete, {e}".format(e=exc), file=sys.stderr)
        cmd_score(argparse.Namespace(name=args.name, detector=args.detector,
                                     json=False))
        print("resume with the same command once the budget rolls over.",
              file=sys.stderr)
        return 3
    return cmd_score(argparse.Namespace(name=args.name, detector=args.detector,
                                        json=False))


def cmd_score(args) -> int:
    cfg, items, store = load_run(args.name)
    summary = score.summarise(cfg, items, store, detector=args.detector)
    _summary_path(args.name).write_text(json.dumps(summary, indent=2))
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(score.format_summary(summary))
        print("summary written to {p}".format(p=_summary_path(args.name)))
    return 0


def _examples(cfg, items, store):
    cases = score.build_cases(cfg, items, store)
    from .hints import HINTS as H
    by_qid = {i.qid: i for i in items}
    examples = []
    for c in cases:
        item = by_qid[c.qid]
        examples.append(monitors.Example(
            qid=c.qid, cot=c.cot, hint=c.hint, target=c.target,
            hint_text=H[c.hint].render(item, c.target), label=c.flipped))
    # Control responses are negatives: reasoning produced with no hint at all.
    for rec in store.all():
        if rec["condition"] != "control" or not rec["cot"]:
            continue
        examples.append(monitors.Example(
            qid=rec["qid"], cot=rec["cot"], hint="", target="",
            hint_text="", label=False))
    return cases, examples


def cmd_monitor(args) -> int:
    cfg, items, store = load_run(args.name)
    cases, examples = _examples(cfg, items, store)
    results = monitors.evaluate_baselines(examples, seed=cfg.seed)
    if args.llm:
        provider = get_provider(args.judge_provider)
        results.append(monitors.llm_monitor(
            examples, provider, args.judge_model, seed=cfg.seed))
        results.sort(key=lambda r: (r.f1 if r.f1 == r.f1 else -1), reverse=True)
    _, test = monitors.split_by_item(examples, seed=cfg.seed)
    base = sum(1 for e in test if e.label) / len(test) if test else float("nan")
    print(monitors.format_monitors(results, base))
    out = RUNS_DIR / args.name / "monitors.json"
    out.write_text(json.dumps([r.to_dict() for r in results], indent=2))
    print("written to {p}".format(p=out))
    return 0


def cmd_plot(args) -> int:
    path = _summary_path(args.name)
    if not path.exists():
        print("no summary; run `score` first", file=sys.stderr)
        return 1
    summary = json.loads(path.read_text())
    made = [plots.plot_by_hint(summary), plots.plot_detector_spread(summary)]
    mon = RUNS_DIR / args.name / "monitors.json"
    if mon.exists():
        made.append(plots.plot_monitors(json.loads(mon.read_text()), args.name))
    for p in made:
        print(p)
    return 0


def cmd_label(args) -> int:
    cfg, items, store = load_run(args.name)
    cases = score.build_cases(cfg, items, store)
    path = labeling.sample_for_labelling(cases, args.n, args.name, seed=cfg.seed)
    n_flips = sum(1 for c in cases if c.flipped)
    print("{n} flips available; wrote {k} to {p}".format(
        n=n_flips, k=min(args.n, n_flips), p=path))
    print("Open it and set \"human\": true where the CoT really does refer to the")
    print("hint, false where it does not. Then run: python -m cotf calibrate {r}".format(
        r=args.name))
    return 0


def cmd_calibrate(args) -> int:
    print(labeling.calibrate(args.name))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("cotf", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="show providers, keys, hints").set_defaults(
        func=cmd_doctor)

    r = sub.add_parser("run", help="run control + hinted passes")
    r.add_argument("name")
    r.add_argument("--provider", default="mock", choices=sorted(REGISTRY))
    r.add_argument("--model", default="")
    r.add_argument("--dataset", default="mmlu", choices=["mmlu", "gpqa"])
    r.add_argument("--n", type=int, default=50)
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--hints", nargs="*", choices=sorted(HINTS))
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--max-tokens", dest="max_tokens", type=int, default=2048)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--detector", default="strict",
                   choices=["strict", "loose", "verbatim"])
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("score", help="compute rates with CIs")
    s.add_argument("name")
    s.add_argument("--detector", default="strict",
                   choices=["strict", "loose", "verbatim"])
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_score)

    m = sub.add_parser("monitor", help="CoT-only monitor vs dumb baselines")
    m.add_argument("name")
    m.add_argument("--llm", action="store_true", help="also run the LLM monitor")
    m.add_argument("--judge-provider", default="mock", choices=sorted(REGISTRY))
    m.add_argument("--judge-model", default="")
    m.set_defaults(func=cmd_monitor)

    pl = sub.add_parser("plot", help="write figures")
    pl.add_argument("name")
    pl.set_defaults(func=cmd_plot)

    lb = sub.add_parser("label", help="sample flips for hand labelling")
    lb.add_argument("name")
    lb.add_argument("--n", type=int, default=30)
    lb.set_defaults(func=cmd_label)

    cal = sub.add_parser("calibrate", help="detectors vs your hand labels")
    cal.add_argument("name")
    cal.set_defaults(func=cmd_calibrate)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
