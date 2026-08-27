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
                 reasoning_body: Optional[Dict[str, Any]] = None):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.env_names = env_names
        self.secret_file = secret_file
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        self.reasoning_body = reasoning_body or {}

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
        for attempt in range(5):
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
                    time.sleep(min(wait, 120))
                    last_err = "HTTP {c}: {t}".format(c=r.status_code, t=r.text[:200])
                    continue
                if r.status_code != 200:
                    return Completion(text="", error="HTTP {c}: {t}".format(
                        c=r.status_code, t=r.text[:400]))
                data = r.json()
                msg = data["choices"][0]["message"]
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
                if not reasoning:
                    content, reasoning = _split_think(content)
                return Completion(text=content.strip(), reasoning=reasoning.strip(),
                                  usage=data.get("usage", {}), raw=data)
            except requests.RequestException as exc:
                last_err = str(exc)
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
        reasoning_body={"reasoning_format": "parsed"}),
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
