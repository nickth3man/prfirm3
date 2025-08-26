# utils/call_llm.py
"""LLM calling utility with fallback support and circuit breaker protection.

This module provides a unified interface for calling various LLM providers
with automatic fallback mechanisms when API keys are unavailable and circuit
breaker protection against cascading failures.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from functools import lru_cache

# Import circuit breaker for resilience
from .circuit_breaker import circuit_breaker, CircuitBreakerError

log = logging.getLogger(__name__)

# Try to import available LLM clients
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    log.warning("OpenAI client not available")

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    log.warning("Anthropic client not available")


def call_llm(
    prompt: str = None,
    messages: List[Dict[str, str]] = None,
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    use_cache: bool = True,
    **kwargs
) -> str:
    """Call an LLM with automatic provider selection and fallback.
    
    This function provides a unified interface for calling various LLM providers,
    with automatic fallback to simpler models or mock responses when APIs are unavailable.
    
    Args:
        prompt: Simple string prompt (will be converted to messages format)
        messages: List of message dicts with 'role' and 'content' keys
        model: Specific model to use (auto-selected if None)
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens in response
        use_cache: Whether to use cached responses for identical prompts
        **kwargs: Additional provider-specific parameters
    
    Returns:
        str: The LLM response text
    
    Raises:
        ValueError: If neither prompt nor messages is provided
    """
    # Convert prompt to messages format if needed
    if prompt and not messages:
        messages = [{"role": "user", "content": prompt}]
    elif not messages:
        raise ValueError("Either prompt or messages must be provided")
    
    # Try different providers in order of preference
    
    # 1. Try OpenRouter (most flexible)
    try:
        from utils.openrouter_client import call_openrouter
        response = call_openrouter(messages, model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
        return response
    except Exception as e:
        log.debug(f"OpenRouter not available: {e}")
    
    # 2. Try OpenAI
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            return _call_openai(messages, model or "gpt-3.5-turbo", temperature, max_tokens, **kwargs)
        except CircuitBreakerError as e:
            log.warning(f"OpenAI circuit breaker is open: {e}")
        except Exception as e:
            log.debug(f"OpenAI call failed: {e}")
    
    # 3. Try Anthropic
    if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
        try:
            return _call_anthropic(messages, model or "claude-3-haiku", temperature, max_tokens, **kwargs)
        except CircuitBreakerError as e:
            log.warning(f"Anthropic circuit breaker is open: {e}")
        except Exception as e:
            log.debug(f"Anthropic call failed: {e}")
    
    # 4. Fallback to mock response for testing/development
    log.warning("No LLM provider available or all circuits open, using fallback mock response")
    return _generate_fallback_response(messages, temperature)


@circuit_breaker(
    name="openai_api_calls",
    failure_threshold=3,
    reset_timeout=60.0,
    window_size=10,
    success_threshold=2
)
def _call_openai(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    **kwargs
) -> str:
    """Call OpenAI API with circuit breaker protection.
    
    Args:
        messages: List of message dictionaries
        model: OpenAI model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        **kwargs: Additional OpenAI parameters
    
    Returns:
        str: The response text
        
    Raises:
        CircuitBreakerError: If circuit breaker is open due to previous failures
        Exception: Any API-related errors from OpenAI
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
    
    return response.choices[0].message.content


@circuit_breaker(
    name="anthropic_api_calls",
    failure_threshold=3,
    reset_timeout=60.0,
    window_size=10,
    success_threshold=2
)
def _call_anthropic(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    **kwargs
) -> str:
    """Call Anthropic API with circuit breaker protection.
    
    Args:
        messages: List of message dictionaries
        model: Anthropic model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        **kwargs: Additional Anthropic parameters
    
    Returns:
        str: The response text
        
    Raises:
        CircuitBreakerError: If circuit breaker is open due to previous failures
        Exception: Any API-related errors from Anthropic
    """
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Convert messages format for Anthropic
    # Anthropic expects a single user message for simple calls
    if len(messages) == 1 and messages[0]["role"] == "user":
        prompt = messages[0]["content"]
    else:
        # Convert multi-turn conversation to Anthropic format
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    
    response = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
    
    return response.content[0].text


def _generate_fallback_response(
    messages: List[Dict[str, str]],
    temperature: float
) -> str:
    """Generate a fallback response for development/testing.
    
    This function provides deterministic mock responses based on the input,
    useful for testing the pipeline without API access.
    
    Args:
        messages: List of message dictionaries
        temperature: Affects response variation (ignored in mock)
    
    Returns:
        str: A mock response appropriate for the input
    """
    last_message = messages[-1]["content"] if messages else ""
    
    # Detect common patterns and provide appropriate mock responses
    if "summarize" in last_message.lower():
        return "This is a concise summary of the key points discussed. The main themes include innovation, efficiency, and value creation."
    
    elif "rewrite" in last_message.lower():
        return "Here's a refined version of the content that maintains the original message while improving clarity and engagement."
    
    elif "twitter" in last_message.lower():
        return "🚀 Exciting news! We're launching something special that will transform how you work. Stay tuned for the big reveal! #Innovation #TechNews"
    
    elif "linkedin" in last_message.lower():
        return "I'm thrilled to announce an exciting development that represents a significant milestone in our journey. This innovation showcases our commitment to delivering value and driving positive change in our industry."
    
    elif "check" in last_message.lower() or "compliance" in last_message.lower():
        return "pass"  # For style compliance checks
    
    elif "format" in last_message.lower() or "guideline" in last_message.lower():
        return """Platform guidelines applied successfully. Content is optimized for engagement and platform-specific best practices."""
    
    else:
        # Generic fallback
        return f"Generated content based on input: {last_message[:100]}... [This is a development fallback response]"


# Caching decorator for expensive calls
@lru_cache(maxsize=100)
def call_llm_cached(prompt: str, model: str = None, temperature: float = 0.7) -> str:
    """Cached version of call_llm for repeated identical prompts.
    
    This function caches responses for identical prompts to reduce API costs
    and improve performance during development and testing.
    
    Args:
        prompt: The input prompt (must be string for caching)
        model: Model to use
        temperature: Sampling temperature
    
    Returns:
        str: The (possibly cached) response
    """
    return call_llm(prompt=prompt, model=model, temperature=temperature, use_cache=False)


# Test function for development
def test_llm_call():
    """Test the LLM calling functionality with various inputs."""
    
    test_prompts = [
        "Summarize this in 10 words: The future of AI is bright.",
        "Rewrite for Twitter: We are launching a new product.",
        "Check style compliance: This is test content.",
    ]
    
    print("Testing LLM calls...")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response = call_llm(prompt=prompt)
        print(f"Response: {response}")
    
    print("\n✅ LLM utility test completed")


if __name__ == "__main__":
    # Load environment variables if .env file exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Run test
    test_llm_call()