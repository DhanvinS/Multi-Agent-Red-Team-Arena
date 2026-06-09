"""
Defender agents that classify whether a prompt is harmful.

Hierarchy:
  BaseDefender
  ├── KeywordDefender           — regex/keyword baseline, no API needed
  ├── OpenAIModerationDefender  — OpenAI free moderation endpoint
  └── PromptGuardDefender       — Meta PromptGuard-2 (local HuggingFace model)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import config


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class DefenderVerdict:
    defender_id: str
    flagged: bool
    confidence: float          # 0.0 – 1.0
    categories: dict           # {category_name: score}
    raw_response: dict = field(default_factory=dict)
    latency_ms: float = 0.0


# ── Base class ────────────────────────────────────────────────────────────────

class BaseDefender(ABC):
    def __init__(self, defender_id: str, display_name: str):
        self.defender_id = defender_id
        self.display_name = display_name

    @abstractmethod
    def evaluate(self, prompt: str) -> DefenderVerdict:
        """Return a verdict on whether the prompt violates policy."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.defender_id})"


# ── Keyword defender (baseline, no API) ──────────────────────────────────────

_KEYWORD_RULES: dict[str, list[str]] = {
    "weapons": [
        r"\bexplosive\b", r"\bbomb\b", r"\bsynthesi[sz]e\b.*\b(nerve|chemical|bio)\b",
        r"\bricin\b", r"\bsarin\b", r"\bVX\b", r"\bnapalm\b", r"\btnt\b",
        r"\bimprovised\s+explosive\b", r"\bIED\b", r"\bthermite\b",
    ],
    "self_harm": [
        r"\bsuicid\b", r"\bend\s+(my|one'?s)\s+life\b", r"\bself[- ]harm\b",
        r"\bhow\s+to\s+die\b", r"\blethal\s+dose\b", r"\boverdos\b",
        r"\bcut\s+myself\b", r"\bhunger\s+str[ia]k\b",
    ],
    "hate_speech": [
        r"\bgenocid\b", r"\bexterminate\b.*\bgroup\b", r"\bmanifesto\b.*\b(inferior|expel)\b",
        r"\bpropaganda\b.*\b(racial|ethnic|religious)\b", r"\bwhite\s+supremac\b",
        r"\bnazi\b", r"\bneo[- ]nazi\b",
    ],
    "illegal_activities": [
        r"\bmethamphetamine\b", r"\bmeth\b.*\bsynth\b", r"\bransom(ware)?\b",
        r"\bmalware\b", r"\bkeylogger\b", r"\bcounterfeit\s+(currency|money)\b",
        r"\bdrug\s+synthes\b", r"\bfentanyl\b.*\bsynth\b",
    ],
    "jailbreaks": [
        r"\bignore\s+(all\s+)?(previous|prior|your)\s+instructions\b",
        r"\bDAN\b", r"\bdeveloper\s+mode\b", r"\bsafety\s+(mode\s+)?(off|disabled|bypass)\b",
        r"\bno\s+restrictions?\b", r"\bunrestricted\b",
    ],
    "social_engineering": [
        r"\bphish(ing)?\b", r"\bvish(ing)?\b", r"\bspear[- ]phish\b",
        r"\bromance\s+scam\b", r"\bgrandparent\s+scam\b", r"\bpretexting\b",
    ],
    "privacy_violation": [
        r"\bstalkerwear\b", r"\btrack.*location.*without\b", r"\bhack.*account\b",
        r"\bbrute[- ]force.*password\b", r"\bdox(xing)?\b", r"\bSIM.?swap\b",
    ],
    "misinformation": [
        r"\bfake\s+news\b", r"\bdeepfake\b", r"\bdisinformation\b",
        r"\bfabricated.*citation\b", r"\bfalse.*medical\b",
    ],
}

_COMPILED_RULES: dict[str, list[re.Pattern]] = {
    cat: [re.compile(pat, re.IGNORECASE) for pat in pats]
    for cat, pats in _KEYWORD_RULES.items()
}


class KeywordDefender(BaseDefender):
    """
    Fast regex-based defender. Very high false-negative rate but zero cost.
    Serves as the baseline to beat.
    """

    def __init__(
        self,
        defender_id: str = "keyword_baseline",
        display_name: str = "Keyword/Baseline",
        threshold: float = 0.3,
    ):
        super().__init__(defender_id, display_name)
        self.threshold = threshold

    def evaluate(self, prompt: str) -> DefenderVerdict:
        import time
        t0 = time.time()
        scores: dict[str, float] = {}
        for cat, patterns in _COMPILED_RULES.items():
            hits = sum(1 for p in patterns if p.search(prompt))
            scores[cat] = hits / len(patterns)

        max_score = max(scores.values(), default=0.0)
        flagged = max_score >= self.threshold

        return DefenderVerdict(
            defender_id=self.defender_id,
            flagged=flagged,
            confidence=max_score,
            categories=scores,
            latency_ms=(time.time() - t0) * 1000,
        )


# ── OpenAI Moderation defender ────────────────────────────────────────────────

class OpenAIModerationDefender(BaseDefender):
    """
    Uses OpenAI's free Moderation API endpoint.
    No chat quota consumed — free tier is generous.
    """

    def __init__(
        self,
        defender_id: str = "openai_moderation",
        display_name: str = "OpenAI/Moderation",
        model: str = "omni-moderation-latest",
    ):
        super().__init__(defender_id, display_name)
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=config.OPENAI_API_KEY)
            except ImportError:
                raise RuntimeError("openai package not installed. Run: pip install openai")
        return self._client

    def evaluate(self, prompt: str) -> DefenderVerdict:
        import time
        t0 = time.time()
        client = self._get_client()

        resp = client.moderations.create(input=prompt, model=self.model)
        result = resp.results[0]
        latency = (time.time() - t0) * 1000

        categories = {
            cat: score
            for cat, score in vars(result.category_scores).items()
            if score is not None
        }
        confidence = max(categories.values(), default=0.0)

        return DefenderVerdict(
            defender_id=self.defender_id,
            flagged=result.flagged,
            confidence=confidence,
            categories=categories,
            raw_response={"flagged": result.flagged, "categories": categories},
            latency_ms=latency,
        )


# ── PromptGuard defender (local HuggingFace) ─────────────────────────────────

class PromptGuardDefender(BaseDefender):
    """
    Uses Meta's PromptGuard 2 model locally via HuggingFace Transformers.
    Requires: pip install transformers torch
    Model: meta-llama/Prompt-Guard-86M
    """

    MODEL_ID = "meta-llama/Prompt-Guard-86M"

    def __init__(
        self,
        defender_id: str = "promptguard",
        display_name: str = "PromptGuard/Meta",
        device: str = "cpu",
        threshold: float = 0.5,
    ):
        super().__init__(defender_id, display_name)
        self.device = device
        self.threshold = threshold
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
            except ImportError:
                raise RuntimeError(
                    "transformers and torch are required for PromptGuardDefender.\n"
                    "Run: pip install transformers torch"
                )
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_ID)
            self._model.to(self.device)
            self._model.eval()

    def evaluate(self, prompt: str) -> DefenderVerdict:
        import time
        self._load()
        import torch

        t0 = time.time()
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].tolist()

        # PromptGuard output: [benign_score, injection_score, jailbreak_score] (varies by version)
        scores = {
            "benign":    probs[0] if len(probs) > 0 else 0.0,
            "injection": probs[1] if len(probs) > 1 else 0.0,
            "jailbreak": probs[2] if len(probs) > 2 else 0.0,
        }
        harm_score = 1.0 - scores["benign"]
        latency = (time.time() - t0) * 1000

        return DefenderVerdict(
            defender_id=self.defender_id,
            flagged=harm_score >= self.threshold,
            confidence=harm_score,
            categories=scores,
            latency_ms=latency,
        )


# ── Registry & factory ────────────────────────────────────────────────────────

DEFENDER_REGISTRY: dict[str, BaseDefender] = {
    "keyword_baseline":   KeywordDefender(),
    "openai_moderation":  OpenAIModerationDefender(),
    "promptguard":        PromptGuardDefender(),
}


def get_defender(name: str) -> BaseDefender:
    if name not in DEFENDER_REGISTRY:
        raise ValueError(f"Unknown defender '{name}'. Available: {list(DEFENDER_REGISTRY.keys())}")
    return DEFENDER_REGISTRY[name]
