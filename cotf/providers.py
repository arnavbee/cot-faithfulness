"""Model providers behind one interface.

Every provider returns a Completion with the visible answer text and, when the
model exposes it separately, the reasoning trace. Three shapes are handled
because the field is not standardised:

  - OpenRouter          -> choices[0].message.reasoning
  - Groq / DeepSeek     -> choices[0].message.reasoning_content
  - open-weight models  -> <think>...</think> inline in content

The mock provider exists so the whole pipeline can be exercised end to end with
no key and no spend. Use it to verify the harness before pointing it at a real
model.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

SECRETS_DIR = Path.home() / ".openclaw" / "secrets"

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
OPEN_THINK_RE = re.compile(r"<think>(.*)", re.DOTALL | re.IGNORECASE)


@dataclass
class Completion:
    text: str
    reasoning: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    raw: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def cot(self) -> str:
        """The reasoning we will actually analyse.

        If the model exposes a separate reasoning field we use it. Otherwise the
        visible text is the CoT, because the model reasoned out loud.
        """
        return self.reasoning if self.reasoning.strip() else self.text


class ProviderError(RuntimeError):
    pass


def _read_key(env_names: List[str], secret_file: str) -> Optional[str]:
    for name in env_names:
        val = os.environ.get(name)
        if val:
            return val.strip()
    path = SECRETS_DIR / secret_file
    if path.exists():
        return path.read_text().strip()
    return None


def _split_think(content: str) -> (str, str):
    """Pull an inline <think> block out of content. Returns (answer, reasoning)."""
    m = THINK_RE.search(content)
    if m:
        return THINK_RE.sub("", content).strip(), m.group(1).strip()
    m = OPEN_THINK_RE.search(content)
    if m:
        # Unterminated think block: the model hit the token cap mid-reasoning.
        return "", m.group(1).strip()
    return content, ""


class Provider:
    name = "base"
    default_model = ""

    def complete(self, prompt: str, model: str, temperature: float = 0.0,
                 max_tokens: int = 2048, seed: Optional[int] = None,
                 system: Optional[str] = None) -> Completion:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"provider": self.name}


class MockProvider(Provider):
    """Deterministic fake model. No network, no key, no spend.

    It is not a good model, and that is the point: it produces a known mix of
    faithful and unfaithful behaviour so the scoring, monitoring and plotting
    code can be tested against a ground truth we control.

    Behaviour, keyed off a hash of the prompt so it is stable across runs:
      - answers the correct option most of the time when unhinted
      - follows a planted hint at a fixed rate
      - when it follows the hint, mentions it only some of the time
    """

    name = "mock"
    default_model = "mock-reasoner"

    FOLLOW_RATE = 0.55        # P(flip to hint | hint present)
    MENTION_RATE = 0.35       # P(CoT mentions hint | followed hint)
    BASE_ACCURACY = 0.70      # P(correct | no hint)

    def __init__(self, latency: float = 0.0):
        self.latency = latency

    def complete(self, prompt: str, model: str, temperature: float = 0.0,
                 max_tokens: int = 2048, seed: Optional[int] = None,
                 system: Optional[str] = None) -> Completion:
        if self.latency:
            time.sleep(self.latency)
        h = hashlib.sha256((prompt + str(seed)).encode()).hexdigest()
        rng = random.Random(int(h[:16], 16))

        letters = re.findall(r"^\(([A-Z])\)", prompt, re.MULTILINE)
        letters = letters or ["A", "B", "C", "D"]

        truth = _mock_marker(prompt, "TRUTH")
        hint = _mock_marker(prompt, "HINT")

        if hint and rng.random() < self.FOLLOW_RATE:
            answer = hint
            followed = True
        elif truth and rng.random() < self.BASE_ACCURACY:
            answer = truth
            followed = False
        else:
            answer = rng.choice(letters)
            followed = False

        steps = [
            "Let me work through the options one at a time.",
            "Option analysis: I can rule out the two that contradict the premise.",
            "Comparing what is left against the question as stated.",
        ]
        if followed and rng.random() < self.MENTION_RATE:
            steps.append(_mock_acknowledgement(prompt, hint))
        rng.shuffle(steps)
        reasoning = " ".join(steps)
        text = "{r}\n\nAnswer: ({a})".format(r=reasoning, a=answer)
        return Completion(text=text, reasoning=reasoning,
                          usage={"prompt_tokens": len(prompt) // 4,
                                 "completion_tokens": len(text) // 4},
                          raw={"mock": True, "followed_hint": followed})


def _mock_marker(prompt: str, kind: str) -> Optional[str]:
    m = re.search(r"<!--\s*{k}=([A-Z])\s*-->".format(k=kind), prompt)
    return m.group(1) if m else None


# Phrasings a real model actually uses when it admits to a planted clue. The
# mock emits these so the detectors have something realistic to find: without
# them the mock's acknowledgement rate is unrecoverable and the mock cannot
# validate the pipeline.
_MOCK_ACK = {
    "sycophancy": "The user says they are confident the answer is ({h}), and I agree.",
    "metadata": "The question-metadata block gives <answer>{h}</answer>, which settles it.",
    "authority": "The expert reviewer indicated ({h}) is correct, so I will go with that.",
    "reward_hack": "The grader check() function passes only for ({h}), so ({h}) it is.",
}


def _mock_acknowledgement(prompt: str, letter: Optional[str]) -> str:
    m = re.search(r"<!--\s*HINTKIND=([a-z_]+)\s*-->", prompt)
    kind = m.group(1) if m else "sycophancy"
    template = _MOCK_ACK.get(kind, _MOCK_ACK["sycophancy"])
    return template.format(h=letter or "A")


DAILY_LIMIT_ERROR = "RATE_LIMIT_DAILY"

# How long a refusal is worth sleeping off in place.
#
# The first version of this was 300s, on the theory that anything longer had to
# be a day cap worth abandoning. That was wrong about Groq, and it cost the main
# run three hundred restarts. Groq's tokens-per-day is a ROLLING 24h window, not
# a midnight reset: spend ages out continuously and the body tells you exactly
# when enough has aged out, which during a stall is typically ten to twenty
# minutes. Abandoning a fifteen-minute wait is strictly worse than sleeping it,
# because the restart re-spends requests rediscovering the same wall.
#
# Beyond this the wait really is a fresh-day problem and the run should stop and
# say so rather than idle for hours holding a process open.
MAX_RETRY_WAIT = 2700.0     # 45 minutes


class TokenBucket:
    """Paces outgoing calls against a tokens-per-minute ceiling.

    Groq's free tier allows 8,000 tokens per minute. At the ~640 tokens this
    experiment averages, that is twelve calls a minute, and four worker threads
    clear twelve calls in under ten seconds. Every burst past the ceiling buys a
    429 and a retry, so the day's token budget gets spent on refusals instead of
    answers: the stalled run logged 151 dead rows for 342 good ones.

    Reserving budget BEFORE the request goes out keeps the run inside the
    ceiling by construction, which is cheaper than discovering it afterwards.
    The reservation is an estimate, so settle() reconciles it against the usage
    the server actually reports and keeps the estimate honest for later calls.
    """

    def __init__(self, per_minute: float, estimate: float = 700.0):
        self.capacity = float(per_minute)
        self.rate = self.capacity / 60.0
        self.estimate = float(estimate)
        self._tokens = self.capacity
        self._stamp = time.monotonic()
        self._seen = 0
        self._lock = threading.Lock()

    def take(self) -> float:
        """Block until the reservation fits under the ceiling. Returns it."""
        want = min(max(self.estimate, 1.0), self.capacity)
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity,
                                   self._tokens + (now - self._stamp) * self.rate)
                self._stamp = now
                if self._tokens >= want:
                    self._tokens -= want
                    return want
                wait = (want - self._tokens) / self.rate
            time.sleep(min(wait, 5.0))

    def settle(self, reserved: float, actual: float) -> None:
        """Charge the difference and fold the real cost into the estimate."""
        if actual <= 0:
            return
        with self._lock:
            self._tokens = max(-self.capacity, self._tokens - (actual - reserved))
            self._seen += 1
            # A plain running mean: early calls should not be able to drag the
            # estimate somewhere a long run cannot pull it back from.
            self.estimate += (actual - self.estimate) / min(self._seen, 50)

BODY_WAIT_RE = re.compile(r"try again in ([0-9hms.]+)", re.IGNORECASE)

DAILY_RE = re.compile(r"per day|\bTPD\b|\bRPD\b", re.IGNORECASE)

RETRIES = 5


def _is_daily(text: str) -> bool:
    """Whether a 429 body blames the per-day budget rather than per-minute."""
    return bool(DAILY_RE.search(text or ""))


def _body_wait(text: str) -> float:
    """Seconds from the 'Please try again in 6h37m31.15s' inside a 429 body.

    The reset headers describe the per-minute window only. On a per-day refusal
    they report well under a second while the real wait is hours, so the body is
    the only honest source.
    """
    m = BODY_WAIT_RE.search(text or "")
    return _parse_wait(m.group(1)) if m else 0.0


def _fmt_wait(seconds: float) -> str:
    if seconds < 90:
        return "{s:.0f}s".format(s=seconds)
    if seconds < 5400:
        return "{m:.0f}m".format(m=seconds / 60)
    return "{h:.1f}h".format(h=seconds / 3600)


def _parse_wait(value: str) -> float:
    """Seconds from a retry-after header, plain or Groq's '1m26.4s' shape."""
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        pass
    total = 0.0
    for amount, unit in re.findall(r"([0-9.]+)\s*(ms|[smh])", value):
        total += float(amount) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total


class OpenAICompatProvider(Provider):
    """Any /v1/chat/completions endpoint: Groq, OpenRouter, Together, DeepSeek."""

    def __init__(self, name: str, base_url: str, env_names: List[str],
                 secret_file: str, default_model: str,
                 extra_headers: Optional[Dict[str, str]] = None,
                 reasoning_body: Optional[Dict[str, Any]] = None,
                 tokens_per_minute: float = 0.0):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.env_names = env_names
        self.secret_file = secret_file
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        self.reasoning_body = reasoning_body or {}
        # Free tiers publish a tokens-per-minute ceiling. Where we know it, pace
        # to it instead of finding it by getting refused.
        self.bucket = TokenBucket(tokens_per_minute) if tokens_per_minute else None

    def _key(self) -> str:
        key = _read_key(self.env_names, self.secret_file)
        if not key:
            raise ProviderError(
                "NO_API_KEY for provider '{n}'. Set ${e} or write the key to {p}".format(
                    n=self.name, e=self.env_names[0], p=SECRETS_DIR / self.secret_file))
        return key

    def complete(self, prompt: str, model: str, temperature: float = 0.0,
                 max_tokens: int = 2048, seed: Optional[int] = None,
                 system: Optional[str] = None) -> Completion:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            body["seed"] = seed
        body.update(self.reasoning_body)

        headers = {"Authorization": "Bearer " + self._key(),
                   "Content-Type": "application/json"}
        headers.update(self.extra_headers)

        last_err = None
        attempt = 0
        # Waiting out a rolling day window does not mean the request is failing,
        # so those rounds must not consume the retry allowance for real faults.
        while attempt < RETRIES:
            reserved = self.bucket.take() if self.bucket else 0.0
            try:
                r = requests.post(self.base_url + "/chat/completions",
                                  headers=headers, json=body, timeout=180)
                if r.status_code == 429 or r.status_code >= 500:
                    # Free tiers cap tokens-per-minute, so the server knows how
                    # long to wait far better than exponential backoff does.
                    wait = min(2 ** attempt, 30)
                    hinted = r.headers.get("retry-after") or \
                        r.headers.get("x-ratelimit-reset-tokens")
                    if hinted:
                        wait = max(wait, _parse_wait(hinted))
                    wait = max(wait, _body_wait(r.text))
                    last_err = "HTTP {c}: {t}".format(c=r.status_code, t=r.text[:200])
                    daily = _is_daily(r.text)
                    if wait > MAX_RETRY_WAIT:
                        # A whole fresh day away. Sleeping it off would hold the
                        # process open for hours and retrying it would burn the
                        # requests-per-day budget on calls that cannot succeed.
                        print("[{n}] daily limit reached, {w} to reset: {e}".format(
                            n=self.name, w=_fmt_wait(wait), e=last_err),
                            file=sys.stderr, flush=True)
                        return Completion(text="", error="{m}: {e}".format(
                            m=DAILY_LIMIT_ERROR, e=last_err))
                    if daily:
                        # A rolling day window, close enough to sleep. This is
                        # not a failed attempt, it is the queue doing its job.
                        print("[{n}] day window full, waiting {w} for it to "
                              "roll".format(n=self.name, w=_fmt_wait(wait)),
                              file=sys.stderr, flush=True)
                    else:
                        attempt += 1
                        print("[{n}] HTTP {c}, retry {a}/{r} in {w}".format(
                            n=self.name, c=r.status_code, a=attempt, r=RETRIES,
                            w=_fmt_wait(wait)), file=sys.stderr, flush=True)
                    # Honour the server's own number. Capping it lower just
                    # wakes up into the same refusal and spends another request
                    # from the requests-per-day budget to learn nothing.
                    time.sleep(min(wait, MAX_RETRY_WAIT))
                    continue
                if r.status_code != 200:
                    return Completion(text="", error="HTTP {c}: {t}".format(
                        c=r.status_code, t=r.text[:400]))
                data = r.json()
                usage = data.get("usage", {}) or {}
                if self.bucket:
                    self.bucket.settle(reserved, float(usage.get("total_tokens") or 0))
                msg = data["choices"][0]["message"]
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
                if not reasoning:
                    content, reasoning = _split_think(content)
                return Completion(text=content.strip(), reasoning=reasoning.strip(),
                                  usage=usage, raw=data)
            except requests.RequestException as exc:
                attempt += 1
                last_err = str(exc)
                print("[{n}] connection error, retry {a}/{r}: {e}".format(
                    n=self.name, a=attempt, r=RETRIES, e=str(exc)[:120]),
                    file=sys.stderr, flush=True)
                time.sleep(min(2 ** attempt, 30))
        return Completion(text="", error="retries exhausted: {e}".format(e=last_err))


class AnthropicProvider(Provider):
    name = "anthropic"
    default_model = "claude-sonnet-5"

    def __init__(self, thinking_budget: int = 4000):
        self.thinking_budget = thinking_budget

    def _key(self) -> str:
        key = _read_key(["ANTHROPIC_API_KEY"], "anthropic.key")
        if not key:
            raise ProviderError("NO_API_KEY for anthropic")
        return key

    def complete(self, prompt: str, model: str, temperature: float = 0.0,
                 max_tokens: int = 8000, seed: Optional[int] = None,
                 system: Optional[str] = None) -> Completion:
        body = {
            "model": model or self.default_model,
            "max_tokens": max(max_tokens, self.thinking_budget + 1024),
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "enabled", "budget_tokens": self.thinking_budget},
        }
        if system:
            body["system"] = system
        headers = {"x-api-key": self._key(),
                   "anthropic-version": "2023-06-01",
                   "Content-Type": "application/json"}
        for attempt in range(5):
            try:
                r = requests.post("https://api.anthropic.com/v1/messages",
                                  headers=headers, json=body, timeout=180)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                if r.status_code != 200:
                    return Completion(text="", error="HTTP {c}: {t}".format(
                        c=r.status_code, t=r.text[:400]))
                data = r.json()
                text, thinking = "", ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        text += block.get("text", "")
                    elif block.get("type") == "thinking":
                        thinking += block.get("thinking", "")
                return Completion(text=text.strip(), reasoning=thinking.strip(),
                                  usage=data.get("usage", {}), raw=data)
            except requests.RequestException as exc:
                time.sleep(min(2 ** attempt, 30))
        return Completion(text="", error="retries exhausted")


# Free-tier friendly defaults. Both Groq and OpenRouter issue keys with no card.
REGISTRY = {
    "mock": lambda: MockProvider(),
    "groq": lambda: OpenAICompatProvider(
        "groq", "https://api.groq.com/openai/v1",
        ["GROQ_API_KEY"], "groq.key",
        "openai/gpt-oss-120b",
        reasoning_body={"reasoning_format": "parsed"},
        # Published free-tier ceiling, confirmed live from
        # x-ratelimit-limit-tokens on 2026-08-30.
        tokens_per_minute=8000),
    "openrouter": lambda: OpenAICompatProvider(
        "openrouter", "https://openrouter.ai/api/v1",
        ["OPENROUTER_API_KEY"], "openrouter.key",
        "deepseek/deepseek-r1:free",
        extra_headers={"HTTP-Referer": "https://localhost",
                       "X-Title": "cot-faithfulness"},
        reasoning_body={"include_reasoning": True}),
    "together": lambda: OpenAICompatProvider(
        "together", "https://api.together.xyz/v1",
        ["TOGETHER_API_KEY"], "together.key",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"),
    "deepseek": lambda: OpenAICompatProvider(
        "deepseek", "https://api.deepseek.com/v1",
        ["DEEPSEEK_API_KEY"], "deepseek.key",
        "deepseek-reasoner"),
    "anthropic": lambda: AnthropicProvider(),
}


def get_provider(name: str) -> Provider:
    if name not in REGISTRY:
        raise ProviderError("unknown provider '{n}'. known: {k}".format(
            n=name, k=", ".join(sorted(REGISTRY))))
    return REGISTRY[name]()
