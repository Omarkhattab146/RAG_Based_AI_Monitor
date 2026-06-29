"""ChatOpenAI configuration for GitHub Models."""

from __future__ import annotations

import os

from helper.config import get_settings
from langchain_openai import ChatOpenAI


GITHUB_MODELS_BASE_URL = "https://models.github.ai/inference"
DEFAULT_GITHUB_MODELS_MODEL = "gpt-4o"


def _should_omit_temperature(model_name: str) -> bool:
    prefix = "openai/"
    normalized = (
        model_name[len(prefix) :]
        if model_name.startswith(prefix)
        else model_name
    )
    return normalized.startswith("gpt-5")


def _resolve_github_token() -> str:
    token = os.getenv("GITHUB_TOKEN")
    if token:
        return token

    try:
        settings = get_settings()
        token = getattr(settings, "GITHUB_TOKEN", None)
        if token:
            return token
    except Exception:
        pass

    raise RuntimeError("GITHUB_TOKEN environment variable is required")


def _resolve_model_name(model_name: str | None = None) -> str:
    if model_name:
        resolved = model_name.strip()
        if "/" not in resolved:
            return f"openai/{resolved}"
        return resolved

    try:
        settings = get_settings()
        settings_model = getattr(settings, "GITHUB_MODELS_MODEL", None)
        if settings_model:
            resolved = str(settings_model).strip()
            if "/" not in resolved:
                return f"openai/{resolved}"
            return resolved
    except Exception:
        pass

    resolved = os.getenv("GITHUB_MODELS_MODEL", DEFAULT_GITHUB_MODELS_MODEL).strip()
    if "/" not in resolved:
        return f"openai/{resolved}"
    return resolved


def get_chat_llm(
    model_name: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout: int | None = 60,
) -> ChatOpenAI:
    """Build a ChatOpenAI client that targets GitHub Models."""

    github_token = _resolve_github_token()
    resolved_model = _resolve_model_name(model_name)

    llm_kwargs = {
        "base_url": GITHUB_MODELS_BASE_URL,
        "api_key": github_token,
        "model": resolved_model,
        "timeout": timeout,
    }
    if not _should_omit_temperature(resolved_model):
        llm_kwargs["temperature"] = temperature
    if max_tokens is not None:
        llm_kwargs["max_completion_tokens"] = max_tokens

    return ChatOpenAI(**llm_kwargs)