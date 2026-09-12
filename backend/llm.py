"""Groq LLM client wrapper (step 1.3).

Single place where the model is created and invoked:

- ``get_llm()``        — lazy singleton ChatGroq client
- ``invoke_llm()``     — invoke with retry + rate-limit backoff
- ``invoke_llm_json()``— structured-output invocation (JSON mode prompt)
- ``extract_json()``   — robust JSON extraction from model text
- ``truncate()``       — prompt-safe text truncation helper

Kept free of any graph/state imports so every worker can depend on it.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq

DEFAULT_MODEL = "openai/gpt-oss-120b"
_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 15


# ---------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------

def get_groq_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Please add it to your .env file."
        )
    return api_key


def create_llm(model: str | None = None, api_key: str | None = None) -> ChatGroq:
    return ChatGroq(
        model=model or DEFAULT_MODEL,
        api_key=api_key or get_groq_api_key(),
    )


_llm: ChatGroq | None = None


def get_llm() -> ChatGroq:
    """Return the shared lazy singleton client."""
    global _llm
    if _llm is None:
        _llm = create_llm()
    return _llm


def set_llm(client: ChatGroq | None) -> None:
    """Override the singleton (tests / custom models)."""
    global _llm
    _llm = client


# ---------------------------------------------------------------------
# Invocation with retry / rate-limit backoff
# ---------------------------------------------------------------------

def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "rate_limit" in text
        or "429" in text
        or "tokens per minute" in text
        or "too many requests" in text
    )


def invoke_llm(
    messages: list[BaseMessage],
    retries: int = _MAX_RETRIES,
    model: str | None = None,
):
    """Invoke the LLM, backing off on rate-limit errors.

    Non-rate-limit errors raise immediately; rate limits retry up to
    ``retries`` attempts with increasing sleep (15s, 30s, ...).
    """
    client = create_llm(model) if model else get_llm()

    for attempt in range(retries):
        try:
            return client.invoke(messages)
        except Exception as exc:
            if _is_rate_limit_error(exc) and attempt < retries - 1:
                time.sleep(_BACKOFF_BASE_SECONDS * (attempt + 1))
                continue
            raise

    raise RuntimeError("invoke_llm: exhausted retries without a response")


# ---------------------------------------------------------------------
# Structured output helpers
# ---------------------------------------------------------------------

def extract_json(text: str) -> Any:
    """Extract a JSON object/array from raw model output.

    Handles ```json fences and surrounding prose. Raises ValueError
    when nothing parseable is found.
    """
    if not text:
        raise ValueError("empty model output")

    # 1. fenced block
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 2. first balanced {...} or [...]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    # 3. whole string fallback
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"no JSON found in model output: {exc}") from exc


def invoke_llm_json(
    messages: list[BaseMessage],
    retries: int = _MAX_RETRIES,
    expect: str = "object",
) -> Any:
    """Invoke the LLM and parse a JSON payload out of the response.

    ``expect`` is only appended to the prompt guidance ("object"/"array").
    """
    response = invoke_llm(messages, retries=retries)
    content = getattr(response, "content", "") or ""
    if isinstance(content, list):  # content-block style responses
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return extract_json(str(content))


# ---------------------------------------------------------------------
# Prompt text helper
# ---------------------------------------------------------------------

def truncate(text: str, max_chars: int = 1500) -> str:
    """Trim text on a word boundary so prompts stay within limits."""
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."
