"""AIQ LLM Client — provider-agnostic LLM abstraction.

Users bring their own API key. The framework provides a simple interface
that works with any provider.

Usage:
    # OpenAI
    llm = create_llm_client("openai", api_key="sk-...")

    # Groq
    llm = create_llm_client("groq", api_key="gsk-...", model="llama-3.3-70b-versatile")

    # Custom function (any provider)
    llm = lambda prompt: my_custom_llm(prompt)

    # Use in pipeline
    config = AIQConfig(llm_provider="openai", llm_api_key="sk-...")
    # or
    config = AIQConfig(llm_call=my_custom_function)
"""
from __future__ import annotations

from typing import Optional, Callable


def create_llm_client(
    provider: str,
    api_key: str = "",
    model: str = "",
    temperature: float = 0.0,
    max_tokens: int = 2000,
) -> Callable[[str], str]:
    """Create an LLM callable for the given provider.

    Args:
        provider: "openai" | "groq" | "anthropic"
        api_key: API key for the provider
        model: model name (uses default if empty)
        temperature: generation temperature (default: 0.0 for deterministic)
        max_tokens: max output tokens (default: 2000)

    Returns:
        callable(prompt: str) -> str
    """
    provider = provider.lower().strip()

    if provider == "openai":
        return _openai_client(api_key, model or "gpt-4o-mini", temperature, max_tokens)
    elif provider == "groq":
        return _groq_client(api_key, model or "llama-3.3-70b-versatile", temperature, max_tokens)
    elif provider == "anthropic":
        return _anthropic_client(api_key, model or "claude-sonnet-4-20250514", temperature, max_tokens)
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Supported: 'openai', 'groq', 'anthropic'. "
            f"Or pass a custom callable via llm_call=my_function."
        )


def _openai_client(api_key: str, model: str, temperature: float,
                   max_tokens: int) -> Callable[[str], str]:
    """Create OpenAI chat completion callable."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "OpenAI SDK not installed. Run: pip install openai"
        )

    client = OpenAI(api_key=api_key)

    def call(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    return call


def _groq_client(api_key: str, model: str, temperature: float,
                 max_tokens: int) -> Callable[[str], str]:
    """Create Groq chat completion callable (OpenAI-compatible API)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "OpenAI SDK not installed (Groq uses OpenAI-compatible API). "
            "Run: pip install openai"
        )

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    def call(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    return call


def _anthropic_client(api_key: str, model: str, temperature: float,
                      max_tokens: int) -> Callable[[str], str]:
    """Create Anthropic chat completion callable."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "Anthropic SDK not installed. Run: pip install anthropic"
        )

    client = anthropic.Anthropic(api_key=api_key)

    def call(prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text or ""

    return call
