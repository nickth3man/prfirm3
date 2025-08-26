import os
import requests
import json
import logging
from typing import List, Dict, Any, Optional, Generator
from datetime import datetime

# Import circuit breaker for resilience
from .circuit_breaker import circuit_breaker, CircuitBreakerError

class OpenRouterClient:
    """Client for OpenRouter API with cost tracking and model routing"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
        # Cost tracking
        self.usage_log = []
        self.cost_rates = {
            'openai/gpt-4o': 0.005,
            'openai/gpt-4o-mini': 0.00015,
            'anthropic/claude-3.5-sonnet': 0.003,
            'anthropic/claude-3-haiku': 0.00025,
            'meta-llama/llama-3.1-8b-instruct': 0.0002,
            'google/gemini-pro': 0.0005,
        }
        
        # Model routing preferences
        self.model_preferences = {
            'creative': 'openai/gpt-4o',
            'analytical': 'anthropic/claude-3.5-sonnet',
            'fast': 'openai/gpt-4o-mini',
            'cost_effective': 'meta-llama/llama-3.1-8b-instruct'
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def call_llm(self, 
                 messages: List[Dict[str, str]], 
                 model: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 stream: bool = False,
                 **kwargs) -> str:
        """
        Call LLM via OpenRouter API
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (auto-selects if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        if not model:
            model = self._select_model(messages, **kwargs)
            
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
            
        if stream:
            return self._stream_response(payload)
        else:
            return self._make_request(payload)

    def _select_model(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Select appropriate model based on content and preferences"""
        # Simple heuristic: check message content for model selection
        content = " ".join([msg.get("content", "") for msg in messages])
        
        if "creative" in kwargs or "creative" in content.lower():
            return self.model_preferences["creative"]
        elif "analytical" in kwargs or "analysis" in content.lower():
            return self.model_preferences["analytical"]
        elif "fast" in kwargs:
            return self.model_preferences["fast"]
        elif "cost_effective" in kwargs:
            return self.model_preferences["cost_effective"]
        else:
            # Default to cost-effective for general use
            return self.model_preferences["cost_effective"]

    @circuit_breaker(
        name="openrouter_api_calls",
        failure_threshold=3,
        reset_timeout=60.0,
        window_size=10,
        success_threshold=2
    )
    def _make_request(self, payload: Dict[str, Any]) -> str:
        """Make non-streaming request to OpenRouter with circuit breaker protection.
        
        Args:
            payload: Request payload for OpenRouter API
            
        Returns:
            str: Response content from the API
            
        Raises:
            CircuitBreakerError: If circuit breaker is open due to previous failures
            requests.exceptions.RequestException: HTTP request errors
        """
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Log usage for cost tracking
            self._log_usage(payload["model"], result)
            
            return content
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"OpenRouter API error: {e}")
            raise

    @circuit_breaker(
        name="openrouter_streaming_calls",
        failure_threshold=3,
        reset_timeout=60.0,
        window_size=10,
        success_threshold=2
    )
    def _stream_response(self, payload: Dict[str, Any]) -> Generator[str, None, None]:
        """Stream response from OpenRouter with circuit breaker protection.
        
        Args:
            payload: Request payload for OpenRouter API
            
        Yields:
            str: Streamed content chunks
            
        Raises:
            CircuitBreakerError: If circuit breaker is open due to previous failures
            requests.exceptions.RequestException: HTTP request errors
        """
        payload["stream"] = True
        
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                stream=True
            )
            response.raise_for_status()
            
            full_content = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and chunk['choices']:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content = delta['content']
                                    full_content += content
                                    yield content
                        except json.JSONDecodeError:
                            continue
            
            # Log usage for cost tracking (approximate)
            self._log_usage(payload["model"], {"usage": {"total_tokens": len(full_content.split())}})
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"OpenRouter streaming error: {e}")
            raise

    def _log_usage(self, model: str, result: Dict[str, Any]):
        """Log API usage for cost tracking"""
        usage = result.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
        
        cost_rate = self.cost_rates.get(model, 0.001)  # Default rate
        cost = (total_tokens / 1000) * cost_rate
        
        usage_entry = {
            "timestamp": datetime.now(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost": cost
        }
        
        self.usage_log.append(usage_entry)
        self.logger.info(f"Usage logged: {model} - {total_tokens} tokens - ${cost:.4f}")

    def get_usage_summary(self, time_period: Optional[datetime] = None) -> Dict[str, Any]:
        """Get usage summary and cost analysis"""
        if time_period:
            filtered_log = [entry for entry in self.usage_log
                          if entry["timestamp"] >= time_period]
        else:
            filtered_log = self.usage_log
            
        total_cost = sum(entry["cost"] for entry in filtered_log)
        total_tokens = sum(entry["total_tokens"] for entry in filtered_log)
        
        model_usage = {}
        for entry in filtered_log:
            model = entry["model"]
            if model not in model_usage:
                model_usage[model] = {
                    "calls": 0,
                    "tokens": 0,
                    "cost": 0
                }
            model_usage[model]["calls"] += 1
            model_usage[model]["tokens"] += entry["total_tokens"]
            model_usage[model]["cost"] += entry["cost"]
            
        return {
            "total_calls": len(filtered_log),
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "model_usage": model_usage,
            "period": time_period
        }

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from OpenRouter"""
        try:
            response = self.session.get(f"{self.base_url}/models")
            response.raise_for_status()
            return response.json()["data"]
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching models: {e}")
            return []

# Global client instance
_client = None

def get_openrouter_client() -> OpenRouterClient:
    """Get or create global OpenRouter client instance"""
    global _client
    if _client is None:
        _client = OpenRouterClient()
    return _client

def call_llm_openrouter(messages: List[Dict[str, str]], 
                       model: Optional[str] = None,
                       temperature: float = 0.7,
                       **kwargs) -> str:
    """Convenience function for LLM calls"""
    client = get_openrouter_client()
    return client.call_llm(messages, model, temperature, **kwargs)

if __name__ == "__main__":
    # Test the client
    client = OpenRouterClient()
    
    messages = [
        {"role": "user", "content": "What is the meaning of life?"}
    ]
    
    try:
        response = client.call_llm(messages, temperature=0.7)
        print(f"Response: {response}")
        
        # Print usage summary
        summary = client.get_usage_summary()
        print(f"Usage Summary: {summary}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure OPENROUTER_API_KEY is set in environment variables")
