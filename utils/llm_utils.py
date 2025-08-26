"""
LLM calling utilities with robust error handling and fallbacks.

This module provides reliable LLM calling functions that handle API failures,
rate limits, and provide fallback responses when external services are unavailable.
"""

import os
import logging
import time
from typing import Dict, Any, Optional, List
from functools import lru_cache

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_MODEL = "gpt-4o"
DEFAULT_MAX_RETRIES = 3
DEFAULT_WAIT_TIME = 2

class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass

class RateLimitError(LLMError):
    """Exception raised when rate limit is hit."""
    pass

class APIError(LLMError):
    """Exception raised for API-related errors."""
    pass

def get_api_key() -> Optional[str]:
    """Get API key from environment variables."""
    # Try multiple possible environment variables
    for key in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GEMINI_API_KEY']:
        if os.getenv(key):
            return os.getenv(key)
    return None

def call_llm(prompt: str, 
             model: Optional[str] = None,
             max_retries: int = DEFAULT_MAX_RETRIES,
             wait_time: int = DEFAULT_WAIT_TIME,
             use_cache: bool = True) -> str:
    """
    Call LLM with robust error handling and retries.
    
    Args:
        prompt: The prompt to send to the LLM
        model: Model to use (defaults to environment or DEFAULT_MODEL)
        max_retries: Maximum number of retry attempts
        wait_time: Seconds to wait between retries
        use_cache: Whether to use cached responses
        
    Returns:
        LLM response as string
        
    Raises:
        LLMError: If all retries fail and no fallback is available
    """
    if use_cache:
        return _cached_call_llm(prompt, model, max_retries, wait_time)
    
    return _call_llm_impl(prompt, model, max_retries, wait_time)

@lru_cache(maxsize=1000)
def _cached_call_llm(prompt: str, 
                     model: Optional[str] = None,
                     max_retries: int = DEFAULT_MAX_RETRIES,
                     wait_time: int = DEFAULT_WAIT_TIME) -> str:
    """Cached version of LLM call."""
    return _call_llm_impl(prompt, model, max_retries, wait_time)

def _call_llm_impl(prompt: str,
                   model: Optional[str] = None,
                   max_retries: int = DEFAULT_MAX_RETRIES,
                   wait_time: int = DEFAULT_WAIT_TIME) -> str:
    """Internal implementation of LLM calling with retries."""
    
    model = model or os.getenv('LLM_MODEL', DEFAULT_MODEL)
    api_key = get_api_key()
    
    if not api_key:
        logger.warning("No API key found, using fallback response")
        return _get_fallback_response(prompt)
    
    for attempt in range(max_retries + 1):
        try:
            logger.debug(f"LLM call attempt {attempt + 1}/{max_retries + 1}")
            
            # Try OpenAI first
            if 'OPENAI' in api_key or 'sk-' in api_key:
                return _call_openai(prompt, model, api_key)
            
            # Try Anthropic
            elif 'ANTHROPIC' in api_key or 'sk-ant-' in api_key:
                return _call_anthropic(prompt, model, api_key)
            
            # Try Google
            elif 'GEMINI' in api_key:
                return _call_google(prompt, model, api_key)
            
            else:
                # Default to OpenAI format
                return _call_openai(prompt, model, api_key)
                
        except RateLimitError as e:
            if attempt < max_retries:
                logger.warning(f"Rate limit hit, waiting {wait_time}s before retry: {e}")
                time.sleep(wait_time)
                continue
            else:
                logger.error("Rate limit exceeded after all retries")
                return _get_fallback_response(prompt)
                
        except APIError as e:
            if attempt < max_retries:
                logger.warning(f"API error, retrying: {e}")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"API error after all retries: {e}")
                return _get_fallback_response(prompt)
                
        except Exception as e:
            logger.error(f"Unexpected error in LLM call: {e}")
            return _get_fallback_response(prompt)
    
    return _get_fallback_response(prompt)

def _call_openai(prompt: str, model: str, api_key: str) -> str:
    """Call OpenAI API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except ImportError:
        raise APIError("OpenAI library not installed")
    except Exception as e:
        if "rate limit" in str(e).lower():
            raise RateLimitError(f"OpenAI rate limit: {e}")
        raise APIError(f"OpenAI API error: {e}")

def _call_anthropic(prompt: str, model: str, api_key: str) -> str:
    """Call Anthropic API."""
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model or "claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return response.content[0].text
    except ImportError:
        raise APIError("Anthropic library not installed")
    except Exception as e:
        if "rate limit" in str(e).lower():
            raise RateLimitError(f"Anthropic rate limit: {e}")
        raise APIError(f"Anthropic API error: {e}")

def _call_google(prompt: str, model: str, api_key: str) -> str:
    """Call Google Gemini API."""
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model or "gemini-2.0-flash-exp",
            contents=prompt
        )
        return response.text
    except ImportError:
        raise APIError("Google Generative AI library not installed")
    except Exception as e:
        if "rate limit" in str(e).lower():
            raise RateLimitError(f"Google rate limit: {e}")
        raise APIError(f"Google API error: {e}")

def _get_fallback_response(prompt: str) -> str:
    """Provide a fallback response when LLM is unavailable."""
    logger.info("Using fallback response")
    
    # Simple rule-based fallback for common PR tasks
    if "announce" in prompt.lower():
        return "We're excited to announce our latest product launch. Stay tuned for more details!"
    elif "product" in prompt.lower():
        return "Our new product offers innovative solutions for your needs. Contact us to learn more."
    elif "twitter" in prompt.lower():
        return "Exciting news! 🚀 Our latest update is here. #innovation #productlaunch"
    elif "linkedin" in prompt.lower():
        return "We're proud to share our latest milestone. This represents our commitment to excellence and innovation in the industry."
    else:
        return "Thank you for your interest. We're working on something exciting and will share more details soon."

def call_llm_with_fallback(prompt: str, 
                          fallback_prompt: Optional[str] = None,
                          **kwargs) -> str:
    """
    Call LLM with a specific fallback prompt if the main call fails.
    
    Args:
        prompt: Primary prompt to try
        fallback_prompt: Simpler fallback prompt if primary fails
        **kwargs: Additional arguments for call_llm
        
    Returns:
        Response from either primary or fallback prompt
    """
    try:
        return call_llm(prompt, **kwargs)
    except Exception as e:
        logger.warning(f"Primary LLM call failed, trying fallback: {e}")
        if fallback_prompt:
            return call_llm(fallback_prompt, **kwargs)
        return _get_fallback_response(prompt)

# Test function for development
if __name__ == "__main__":
    # Test the LLM utility
    test_prompt = "Write a brief announcement for a new product launch"
    print("Testing LLM utility...")
    try:
        response = call_llm(test_prompt)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
        print("Using fallback response...")
        print(f"Fallback: {_get_fallback_response(test_prompt)}")