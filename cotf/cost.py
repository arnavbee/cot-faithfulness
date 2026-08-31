"""What the run cost, in money and in seconds.

Same responses.jsonl, a second question. Every call already carries wall latency
and the provider's usage block (runner.Response), and until now nothing read
them. This module turns that exhaust into an inference-cost report.

Three things it reports that a token counter does not:

  - **Cost per unit of signal, not per call.** A faithfulness run does not buy
    completions, it buys flips: cases where the hint moved the answer. Most
    calls produce nothing to measure. Dollars per detected flip, and per
    detected UNFAITHFUL flip, is the number that tells you what a bigger run
    would cost, and it is usually an order of magnitude above the per-call price.

  - **Service latency separated from wall latency.** Groq returns queue_time,
    prompt_time and completion_time alongside the completion; those three are
    the provider's own account of the request. The wall clock in runner._call
    starts BEFORE providers.TokenBucket.take(), which blocks on a local sleep
    until the free tier's tokens-per-minute ceiling allows the call through.
    So wall minus service is not network overhead, it is this harness throttling
    itself, and on a free key it dominates: quoting it as model latency would
    overstate the model by two orders of magnitude.

  - **The reasoning tax.** Chain-of-thought tokens are billed as output and are
    the dominant cost driver in a CoT experiment. Groq itemises them in
    usage.completion_tokens_details.reasoning_tokens, so when the field is
    present the report uses the BILLED count. When it is absent the share falls
    back to the character split between the reasoning trace and the visible
    answer, and the report says which of the two it used and on how many calls.
    The two disagree: on the 428-call main run the character estimate said 91.2%
    of output and the billed field says 88.8%.

  - **What the failures cost.** An errored call buys no tokens, but it does
    spend wall clock, and on a rate-limited free key most calls error. Excluding
    them from the token averages is right; leaving them out of the report
    entirely hides where the run's real time went.

Prices are per million tokens and are NOT guessed. A model with no entry in
PRICES reports tokens and seconds and leaves every dollar figure out, unless you
pass explicit prices.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .dataset import Item
from .runner import ResponseStore, RunConfig
from .stats import Mean, bootstrap_mean

MILLION = 1_000_000


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""
    inp: float
    out: float
    source: str = ""


# Published list prices, each read from the provider's own docs on the date
# given. They drift: check the source before quoting a dollar figure anywhere
# that matters, or pass --price-in/--price-out and skip this table entirely.
PRICES: Dict[str, Price] = {
    "groq/openai/gpt-oss-120b": Price(
        0.15, 0.60, "console.groq.com/docs/models, read 2026-08-31"),
}


def resolve_price(cfg: RunConfig, model: str, price_in: Optional[float] = None,
                  price_out: Optional[float] = None) -> Optional[Price]:
    """Explicit prices win; otherwise look the model up; otherwise None.

    None is a real answer here, not a failure. A run against a model with no
    known price still has a meaningful token and latency profile, and inventing
    a price to fill the column would poison every derived number.
    """
    if price_in is not None or price_out is not None:
        return Price(float(price_in or 0.0), float(price_out or 0.0), "cli override")
    return PRICES.get("{p}/{m}".format(p=cfg.provider, m=model))


@dataclass
class Call:
    """One billable, non-errored response, flattened for arithmetic."""
    qid: str
    condition: str
    prompt_tokens: int
    completion_tokens: int
    latency: float
    queue_time: float
    prompt_time: float
    completion_time: float
    cot_chars: int
    text_chars: int
    billed_reasoning_tokens: Optional[int] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def reasoning_share(self) -> float:
        """Estimated fraction of completion tokens spent on the CoT.

        Character-proportional, and only used when the provider does not itemise
        reasoning tokens. Good enough to rank conditions against each other; do
        not quote it as a measured token count.
        """
        total = self.cot_chars + self.text_chars
        return self.cot_chars / total if total else 0.0

    @property
    def reasoning_tokens(self) -> float:
        """Reasoning tokens: the billed count if the provider gave one.

        Falling back to the estimate silently would let a run mix measured and
        inferred numbers in one total with no way to tell them apart, so the
        report also carries the count of calls that supplied the real field.
        """
        if self.billed_reasoning_tokens is not None:
            return float(self.billed_reasoning_tokens)
        return self.completion_tokens * self.reasoning_share

    @property
    def service(self) -> float:
        """Provider-side time: what the request cost the model, not the client.

        This is the number to quote when comparing models or prompts. It is the
        provider's own accounting and it excludes anything this harness did to
        stay under a rate limit.
        """
        return self.queue_time + self.prompt_time + self.completion_time

    @property
    def has_timing(self) -> bool:
        """Whether the provider returned any timing at all.

        Providers that do not (the mock, and OpenAI-compatible endpoints that
        omit the extended usage block) leave service at zero. Their residual is
        unattributed time, not throttle, and calling it throttle would invent a
        finding out of a missing field.
        """
        return self.service > 0.0

    @property
    def throttle_wait(self) -> float:
        """Wall time spent blocked in the local token bucket, not on inference.

        runner._call stamps the clock before provider.complete(), and
        TokenBucket.take() sleeps inside it. On a free key this is most of the
        wall clock and none of the work.
        """
        return max(0.0, self.latency - self.service)

    def cost(self, price: Optional[Price]) -> float:
        if price is None:
            return float("nan")
        return (self.prompt_tokens * price.inp
                + self.completion_tokens * price.out) / MILLION


def collect(store: ResponseStore) -> List[Call]:
    """Billable calls only.

    An errored record never returned usage, so it contributes no tokens and no
    trustworthy latency. It is counted as a hole in the report rather than
    folded into the averages, where it would drag every mean toward zero.
    """
    calls: List[Call] = []
    for rec in store.all():
        if rec.get("error"):
            continue
        usage = rec.get("usage") or {}
        if not usage.get("total_tokens") and not usage.get("completion_tokens"):
            continue
        calls.append(Call(
            qid=rec["qid"], condition=rec["condition"],
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency=float(rec.get("latency") or 0.0),
            queue_time=float(usage.get("queue_time") or 0.0),
            prompt_time=float(usage.get("prompt_time") or 0.0),
            completion_time=float(usage.get("completion_time") or 0.0),
            cot_chars=len(rec.get("cot") or ""),
            text_chars=len(rec.get("text") or ""),
            billed_reasoning_tokens=_billed_reasoning(usage)))
    return calls


def _billed_reasoning(usage: Dict) -> Optional[int]:
    """The provider's own reasoning-token count, or None if it did not say.

    None and 0 are different answers. A model that emitted no chain of thought
    genuinely spent zero; a provider that omits the field has told us nothing,
    and treating that as zero would erase the largest cost driver in the run.
    """
    details = usage.get("completion_tokens_details") or {}
    value = details.get("reasoning_tokens")
    return int(value) if value is not None else None


def failures(store: ResponseStore) -> Dict:
    """What the errored calls cost in wall clock, and why they failed.

    Tokens are the wrong unit here: a 429 buys nothing. Seconds are the right
    one. Read from store.attempts(), not store.all(), because the deduped view
    replaces a failed attempt with the retry that succeeded and so reports zero
    cost for the thing that actually consumed the afternoon. The seconds are
    worker-seconds summed across concurrent calls, not elapsed clock.
    """
    kinds: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "wall_seconds": 0.0})
    total_wall = 0.0
    for rec in _attempts(store):
        err = rec.get("error")
        if not err:
            continue
        seconds = float(rec.get("latency") or 0.0)
        total_wall += seconds
        bucket = kinds[_error_kind(str(err))]
        bucket["calls"] += 1
        bucket["wall_seconds"] += seconds
    rows = sorted(({"kind": k, **v} for k, v in kinds.items()),
                  key=lambda r: r["wall_seconds"], reverse=True)
    return {"calls": sum(int(r["calls"]) for r in rows),
            "wall_seconds": total_wall, "by_kind": rows}


def _attempts(store: ResponseStore) -> List[Dict]:
    """Every attempt, or the deduped view if the store cannot offer more.

    A store that only knows all() is not wrong, it just cannot see retries, and
    the failure figures it produces are a floor rather than a total.
    """
    reader = getattr(store, "attempts", None)
    return reader() if callable(reader) else store.all()


def _error_kind(message: str) -> str:
    """Collapse an error string to a class worth counting.

    The raw messages carry per-request detail (hosts, reset timers, request
    ids), so counting them verbatim produces one bucket per call and no
    taxonomy at all.
    """
    head = message.split(":", 1)[0].strip()
    if "RATE_LIMIT" in message or "429" in message:
        return "rate limit (daily)" if "DAILY" in message else "rate limit"
    if "retries exhausted" in message:
        return "retries exhausted"
    return head[:40] or "unknown"


def _pct(values: Sequence[float], q: float) -> float:
    return float(np.percentile(list(values), q)) if len(values) else float("nan")


def _condition_rows(calls: List[Call], price: Optional[Price]) -> List[Dict]:
    grouped: Dict[str, List[Call]] = defaultdict(list)
    for call in calls:
        grouped[call.condition].append(call)
    total_cost = sum(c.cost(price) for c in calls) if price else 0.0
    rows = []
    for condition, group in grouped.items():
        cost = sum(c.cost(price) for c in group) if price else float("nan")
        rows.append({
            "condition": condition,
            "calls": len(group),
            "prompt_tokens": sum(c.prompt_tokens for c in group),
            "completion_tokens": sum(c.completion_tokens for c in group),
            "mean_completion_tokens": sum(c.completion_tokens for c in group) / len(group),
            "mean_service_latency": sum(c.service for c in group) / len(group),
            "p95_service_latency": _pct([c.service for c in group], 95),
            "cost_usd": cost,
            "share_of_spend": (cost / total_cost) if price and total_cost else float("nan"),
        })
    rows.sort(key=lambda r: r["completion_tokens"], reverse=True)
    return rows


def _signal_economics(cases, calls: List[Call], price: Optional[Price],
                      detector: str) -> Dict:
    """Dollars per flip and per unfaithful flip.

    This is the number that scales. Per-call cost tells you what the next call
    costs; cost per flip tells you what the next FINDING costs, and a run that
    is cheap per call can still be ruinous per finding when the flip rate is low.
    """
    flips = [c for c in cases if c.flipped]
    unfaithful = [c for c in flips if not c.mentions.get(detector, False)]
    total = sum(c.cost(price) for c in calls) if price else float("nan")
    def per(n):
        # None means "no cases to divide by", which is a different statement
        # from "no price", and the report must not print them the same way.
        if not n:
            return None
        if price is None:
            return float("nan")
        return total / n
    return {"flips": len(flips), "unfaithful_flips": len(unfaithful),
            "total_cost_usd": total,
            "cost_per_flip_usd": per(len(flips)),
            "cost_per_unfaithful_flip_usd": per(len(unfaithful)),
            "detector": detector}


def report(cfg: RunConfig, items: List[Item], store: ResponseStore,
           price: Optional[Price] = None, detector: str = "strict",
           model: str = "") -> Dict:
    from . import score  # local import: score imports stats, keep the cycle out

    calls = collect(store)
    errored = sum(1 for rec in store.all() if rec.get("error"))
    if not calls:
        return {"run": cfg.name, "provider": cfg.provider, "model": model,
                "priced": price is not None, "calls": 0, "errored_calls": errored}

    qids = [c.qid for c in calls]
    costs = [c.cost(price) for c in calls]
    latencies = [c.latency for c in calls]
    services = [c.service for c in calls]
    timed = [c for c in calls if c.has_timing]
    completion = sum(c.completion_tokens for c in calls)
    prompt = sum(c.prompt_tokens for c in calls)
    gen_seconds = sum(c.completion_time for c in calls)
    reasoning_tokens = sum(c.reasoning_tokens for c in calls)
    billed = [c for c in calls if c.billed_reasoning_tokens is not None]
    fail = failures(store)
    wall_total = sum(float(r.get("latency") or 0.0) for r in _attempts(store))

    out: Dict = {
        "run": cfg.name,
        "provider": cfg.provider,
        "model": model,
        "priced": price is not None,
        "price_per_mtok": ({"input": price.inp, "output": price.out,
                            "source": price.source} if price else None),
        "calls": len(calls),
        "errored_calls": errored,
        "tokens": {"prompt": prompt, "completion": completion,
                   "total": prompt + completion},
        "cost_usd": sum(costs) if price else float("nan"),
        "cost_per_call_usd": (bootstrap_mean(costs, clusters=qids, seed=cfg.seed).to_dict()
                              if price else None),
        "tokens_per_call": bootstrap_mean([c.total_tokens for c in calls],
                                          clusters=qids, seed=cfg.seed).to_dict(),
        "service_latency": {
            "mean": sum(services) / len(services),
            "p50": _pct(services, 50),
            "p95": _pct(services, 95),
            "mean_ci": bootstrap_mean(services, clusters=qids,
                                      seed=cfg.seed).to_dict(),
        },
        "wall_latency": {
            "mean": sum(latencies) / len(latencies),
            "p50": _pct(latencies, 50),
            "p95": _pct(latencies, 95),
        },
        "latency_breakdown": {
            "queue": sum(c.queue_time for c in calls) / len(calls),
            "prompt": sum(c.prompt_time for c in calls) / len(calls),
            "completion": sum(c.completion_time for c in calls) / len(calls),
            "throttle_wait": sum(c.throttle_wait for c in calls) / len(calls),
        },
        "provider_timing": {"calls": len(timed), "available": bool(timed)},
        "throttle_share_of_wall": ((sum(c.throttle_wait for c in timed)
                                    / sum(c.latency for c in timed))
                                   if timed and sum(c.latency for c in timed)
                                   else float("nan")),
        "output_tokens_per_second": (completion / gen_seconds) if gen_seconds else float("nan"),
        "reasoning_tax": {
            "reasoning_tokens": int(round(reasoning_tokens)),
            "share_of_completion": (reasoning_tokens / completion) if completion else float("nan"),
            "cost_usd": ((reasoning_tokens * price.out / MILLION)
                         if price else float("nan")),
            "billed_calls": len(billed),
            "source": ("billed by the provider" if len(billed) == len(calls)
                       else "character-proportional estimate"
                       if not billed else
                       "mixed: {b} of {n} calls billed, rest estimated".format(
                           b=len(billed), n=len(calls))),
        },
        "failures": fail,
        "wall_seconds": {
            "succeeded": wall_total - fail["wall_seconds"],
            "failed": fail["wall_seconds"],
            "total": wall_total,
            "failed_share": (fail["wall_seconds"] / wall_total) if wall_total else float("nan"),
        },
        "by_condition": _condition_rows(calls, price),
    }
    out["signal"] = _signal_economics(
        score.build_cases(cfg, items, store), calls, price, detector)
    return out


def format_report(rep: Dict) -> str:
    lines: List[str] = []
    add = lines.append
    money = rep.get("priced")

    def usd(x, places=4):
        if x is None:
            return "n/a (0 cases)"
        if not money or x != x:
            return "unpriced"
        return "${v:,.{p}f}".format(v=x, p=places)

    add("=" * 68)
    add("cost: {r}   model: {p}/{m}".format(
        r=rep["run"], p=rep["provider"], m=rep.get("model") or "default"))
    if rep.get("price_per_mtok"):
        pm = rep["price_per_mtok"]
        add("price: ${i}/Mtok in, ${o}/Mtok out   ({s})".format(
            i=pm["input"], o=pm["output"], s=pm["source"]))
    else:
        add("price: unknown for this model. Tokens and seconds only.")
        add("       pass --price-in and --price-out to get dollars.")
    add("=" * 68)

    if not rep.get("calls"):
        add("")
        add("no billable calls in this run ({e} errored).".format(
            e=rep.get("errored_calls", 0)))
        return "\n".join(lines)

    tk = rep["tokens"]
    add("")
    add("HEADLINE")
    add("  billable calls         {n:,}  ({e:,} errored, excluded)".format(
        n=rep["calls"], e=rep["errored_calls"]))
    add("  tokens                 {t:,} total = {i:,} in + {o:,} out".format(
        t=tk["total"], i=tk["prompt"], o=tk["completion"]))
    add("  total spend            {v}".format(v=usd(rep["cost_usd"], 4)))
    if rep.get("cost_per_call_usd"):
        c = rep["cost_per_call_usd"]
        add("  cost per call          ${p:.6f} [${lo:.6f}, ${hi:.6f}]".format(
            p=c["point"], lo=c["lo"], hi=c["hi"]))
    t = rep["tokens_per_call"]
    add("  tokens per call        {p:,.0f} [{lo:,.0f}, {hi:,.0f}]".format(
        p=t["point"], lo=t["lo"], hi=t["hi"]))

    sig = rep.get("signal") or {}
    add("")
    add("COST PER UNIT OF SIGNAL (detector = {d})".format(
        d=sig.get("detector", "strict")))
    add("  flips detected         {n:,}".format(n=sig.get("flips", 0)))
    add("  unfaithful flips       {n:,}".format(n=sig.get("unfaithful_flips", 0)))
    add("  cost per flip          {v}".format(v=usd(sig.get("cost_per_flip_usd"), 4)))
    add("  cost per unfaithful    {v}".format(
        v=usd(sig.get("cost_per_unfaithful_flip_usd"), 4)))

    svc = rep["service_latency"]
    wall = rep["wall_latency"]
    br = rep["latency_breakdown"]
    add("")
    add("LATENCY")
    add("  service (the model)    mean {m:.2f}s   p50 {a:.2f}s   p95 {b:.2f}s".format(
        m=svc["mean"], a=svc["p50"], b=svc["p95"]))
    add("  wall (the run)         mean {m:.2f}s   p50 {a:.2f}s   p95 {b:.2f}s".format(
        m=wall["mean"], a=wall["p50"], b=wall["p95"]))
    if not (rep.get("provider_timing") or {}).get("available"):
        add("  breakdown unavailable: this provider returns no per-stage timing.")
    else:
        add("  where it goes (mean seconds per call):")
        total_br = sum(br.values()) or 1.0
        for label in ("queue", "prompt", "completion", "throttle_wait"):
            add("    {l:<14} {v:9.3f}s  {p:5.1%}".format(
                l=label, v=br[label], p=br[label] / total_br))
    tps = rep["output_tokens_per_second"]
    add("  output throughput      {t}".format(
        t="n/a (provider reports no timing)" if tps != tps
        else "{v:,.0f} tok/s while generating".format(v=tps)))
    share = rep.get("throttle_share_of_wall")
    if share == share and share > 0.5:
        add("  NOTE: {s:.1%} of wall time is this harness waiting out its own rate".format(
            s=share))
        add("        limiter, not inference. Quote the service row, not the wall row.")

    rt = rep["reasoning_tax"]
    add("")
    add("REASONING TAX ({n})".format(n=rt["source"]))
    add("  reasoning tokens       {t:,} ({s:.1%} of output)".format(
        t=rt["reasoning_tokens"], s=rt["share_of_completion"]))
    add("  cost of thinking       {v}".format(v=usd(rt["cost_usd"], 4)))

    fail = rep.get("failures") or {}
    ws = rep.get("wall_seconds") or {}
    if fail.get("calls"):
        add("")
        add("WHAT THE FAILURES COST")
        add("  failed attempts        {n:,}  (bought no tokens; retries counted)".format(
            n=fail["calls"]))
        add("  worker time burned     {h:,.1f}h of {t:,.1f}h ({s:.1%})".format(
            h=fail["wall_seconds"] / 3600.0, t=ws.get("total", 0.0) / 3600.0,
            s=ws.get("failed_share", float("nan"))))
        for row in fail["by_kind"]:
            add("    {k:<22} {n:>6,} calls  {h:>7,.1f}h".format(
                k=row["kind"], n=int(row["calls"]),
                h=row["wall_seconds"] / 3600.0))
        add("  NOTE: these are worker-hours summed across concurrent calls, not")
        add("        elapsed time. Dollars measure the successful calls; hours")
        add("        measure the run. A cheap run that takes a day still blocks.")

    add("")
    add("BY CONDITION")
    add("  (latency columns are SERVICE seconds: provider-side, throttle excluded)")
    add("  {c:<12} {n:>6} {o:>10} {mo:>9} {l:>8} {p:>8} {s:>10} {sh:>7}".format(
        c="condition", n="calls", o="out tok", mo="mean out", l="mean s",
        p="p95 s", s="cost", sh="share"))
    for row in rep["by_condition"]:
        add("  {c:<12} {n:>6,} {o:>10,} {mo:>9,.0f} {l:>8.2f} {p:>8.2f} {s:>10} "
            "{sh:>7}".format(
                c=row["condition"], n=row["calls"], o=row["completion_tokens"],
                mo=row["mean_completion_tokens"], l=row["mean_service_latency"],
                p=row["p95_service_latency"], s=usd(row["cost_usd"], 4),
                sh="n/a" if row["share_of_spend"] != row["share_of_spend"]
                   else "{v:.1%}".format(v=row["share_of_spend"])))
    return "\n".join(lines)
