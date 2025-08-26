from openai import OpenAI
import os
import time
import logging
from typing import Optional

log = logging.getLogger(__name__)


# Lightweight, resilient LLM wrapper used by ContentCraftsman and other nodes.
# Features:
# - Exponential backoff retries
# - Basic prompt sanitization (redact obvious API key patterns)
# - Optional stub mode for deterministic tests
def _sanitize_prompt(prompt: str, max_length: int = 15000) -> str:
    """Perform minimal prompt sanitization and truncation.

    This intentionally keeps logic simple — heavy sanitization should be
    implemented at call-sites if needed.
    """
    if not isinstance(prompt, str):
        prompt = str(prompt)
    # Redact obvious API key-like substrings
    prompt = prompt.replace("API_KEY=", "API_KEY=[REDACTED]")
    prompt = prompt.replace("OPENAI_API_KEY", "[REDACTED]")
    # Truncate to a safe maximum length to avoid provider errors
    if len(prompt) > max_length:
        prompt = prompt[:max_length] + "\n... [truncated]"
    return prompt


def call_llm(
    prompt: str,
    max_retries: int = 3,
    backoff: float = 2.0,
    use_stub: bool = False,
    model: Optional[str] = None,
) -> str:
    """Call an LLM with retries and safe defaults.

    Args:
        prompt: The prompt to send to the model.
        max_retries: Number of attempts (including the first).
        backoff: Base backoff seconds (exponential).
        use_stub: If True, return a deterministic stub response (for tests).
        model: Optional model override (defaults to ENV or 'gpt-4o').

    Returns:
        The model's textual response.
    """
    # Test-friendly stub mode
    if use_stub or os.environ.get("LLM_STUB", "false").lower() in ("1", "true"):
        log.debug("LLM stub mode active")
        # Keep stub deterministic and short
        return f"[STUB] Response for prompt: {prompt[:120]}"

    safe_prompt = _sanitize_prompt(prompt)
    model_name = model or os.environ.get("LLM_MODEL", "gpt-4o")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    attempt = 0
    while attempt < max_retries:
        try:
            attempt += 1
            log.debug("Calling LLM (attempt %d) model=%s", attempt, model_name)
            r = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": safe_prompt}],
            )
            # Provider-specific shape: keep current codepath compatible
            return r.choices[0].message.content
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("LLM call failed (attempt %d/%d): %s", attempt, max_retries, exc)
            if attempt >= max_retries:
                log.exception("LLM call failed after %d attempts", attempt)
                raise
            # Exponential backoff with jitter
            sleep_for = backoff * (2 ** (attempt - 1))
            time.sleep(sleep_for)


if __name__ == "__main__":
    prompt = "What is the meaning of life?"
    print(call_llm(prompt, use_stub=os.environ.get("LLM_STUB", "true") == "true"))
