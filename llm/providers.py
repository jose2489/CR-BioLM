"""
Provider resolution for multimodal LLM calls.

Maps a MODEL SPEC to an OpenAI-compatible chat-completions endpoint, so the same
payload (text + base64 ``image_url`` parts) works against a hosted API or a local
model with no change to the caller.

Spec format — ``"<provider>:<model>"``:

    "openrouter:openai/gpt-4o"
    "openrouter:anthropic/claude-sonnet-4-5"
    "ollama:qwen3-vl:8b-instruct"          <- model name itself contains ':'

A bare spec with no known provider prefix (e.g. ``"openai/gpt-4o"``) is treated as
OpenRouter, so existing call sites keep working.

Environment overrides:
    LLM_PROVIDER      default provider for bare specs        (default: openrouter)
    OLLAMA_BASE_URL   local endpoint                          (default: http://localhost:11434/v1)
    OPENROUTER_API_KEY

Ollama note: the ``/v1`` OpenAI-compatible endpoint IGNORES ``options.num_ctx``.
Set ``OLLAMA_CONTEXT_LENGTH`` in the environment and restart the Ollama service, or a
long prompt with two images is silently truncated. Verify with ``ollama ps`` — the
CONTEXT column must show the value you set, not 4096.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_OLLAMA_DEFAULT = "http://localhost:11434/v1"


@dataclass(frozen=True)
class Provider:
    """Everything needed to issue one chat-completions request."""
    name: str            # "openrouter" | "ollama"
    url: str             # full chat-completions endpoint
    model: str           # provider-native model id
    api_key: str | None  # None when the provider needs no auth
    is_local: bool

    @property
    def spec(self) -> str:
        return f"{self.name}:{self.model}"

    def headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h


def _ollama_url() -> str:
    base = os.environ.get("OLLAMA_BASE_URL", _OLLAMA_DEFAULT).rstrip("/")
    return base if base.endswith("/chat/completions") else f"{base}/chat/completions"


def resolve(spec: str, *, api_key: str | None = None) -> Provider:
    """Resolve a model spec to a Provider.

    ``api_key`` overrides the environment (used by callers that already hold a key).
    """
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("empty model spec")

    # Split on the FIRST colon only — Ollama model ids contain colons
    # ("qwen3-vl:8b-instruct"), so a naive split would mangle them.
    provider, _, model = spec.partition(":")
    provider = provider.lower()

    if provider not in ("openrouter", "ollama") or not model:
        # Bare spec such as "openai/gpt-4o" — use the default provider.
        provider = os.environ.get("LLM_PROVIDER", "openrouter").lower()
        model = spec

    if provider == "ollama":
        return Provider(
            name="ollama", url=_ollama_url(), model=model,
            api_key=None, is_local=True,
        )

    return Provider(
        name="openrouter", url=_OPENROUTER_URL, model=model,
        api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
        is_local=False,
    )


def is_configured(spec: str, *, api_key: str | None = None) -> tuple[bool, str]:
    """Can this spec actually be called? Returns (ok, reason_if_not)."""
    try:
        p = resolve(spec, api_key=api_key)
    except ValueError as e:
        return False, str(e)
    if p.name == "openrouter" and not p.api_key:
        return False, "OPENROUTER_API_KEY not set"
    return True, ""


def slug(spec: str) -> str:
    """Filesystem-safe token for a spec (used in output filenames)."""
    return (spec.replace("/", "_").replace("-", "_")
                .replace(":", "_").replace(" ", "_"))
