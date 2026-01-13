"""
LLM Provider Layer - Robust interface for LLM calls with retry and caching.

Provides:
- Retry with exponential backoff on rate limits (429/503)
- Response caching by (text_hash, prompt_hash, model_id)
- Failure artifacts with error codes
- Provider abstraction (Gemini/OpenAI/local)

Usage:
    from harness.llm_provider import get_llm_completion, LLMResult
    
    result = get_llm_completion(
        prompt="Summarize: ...",
        model="gemini-2.0-flash",
        text_hash="abc123",   # For cache key
        prompt_hash="def456",  # For cache key
    )
    
    if result.success:
        print(result.text)
    else:
        print(f"Failed: {result.error_code}")
"""

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path("runs/llm_cache")

# Error codes for failure artifacts
class ErrorCode:
    RATE_LIMITED = "RATE_LIMITED"
    API_ERROR = "API_ERROR"
    TIMEOUT = "TIMEOUT"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    INVALID_RESPONSE = "INVALID_RESPONSE"


@dataclass
class LLMResult:
    """Result of an LLM call."""
    success: bool
    text: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    model_id: str = ""
    latency_ms: float = 0
    cached: bool = False
    attempts: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'text': self.text,
            'error_code': self.error_code,
            'error_message': self.error_message,
            'model_id': self.model_id,
            'latency_ms': round(self.latency_ms, 1),
            'cached': self.cached,
            'attempts': self.attempts,
        }


def compute_cache_key(text_hash: str, prompt_hash: str, model_id: str) -> str:
    """Compute cache key from inputs."""
    combined = f"{text_hash}:{prompt_hash}:{model_id}"
    return hashlib.sha256(combined.encode()).hexdigest()[:24]


def get_cached_response(cache_key: str) -> Optional[str]:
    """Get cached response if exists."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                data = json.load(f)
                return data.get('response')
        except:
            pass
    return None


def save_to_cache(cache_key: str, response: str, model_id: str) -> None:
    """Save response to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    data = {
        'response': response,
        'model_id': model_id,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(cache_file, 'w') as f:
        json.dump(data, f, indent=2)


def call_gemini(prompt: str, model: str = "gemini-2.0-flash") -> Tuple[str, Dict[str, Any]]:
    """
    Call Gemini API.
    
    Returns:
        (response_text, metrics)
        
    Raises:
        Rate limit or API errors
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("google-generativeai not installed")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    
    genai.configure(api_key=api_key)
    
    t0 = time.time()
    model_obj = genai.GenerativeModel(model)
    response = model_obj.generate_content(prompt)
    latency_ms = (time.time() - t0) * 1000
    
    metrics = {
        'latency_ms': latency_ms,
        'model': model,
    }
    
    return response.text.strip(), metrics


def call_openai(prompt: str, model: str = "gpt-4o-mini") -> Tuple[str, Dict[str, Any]]:
    """
    Call OpenAI API.
    
    Returns:
        (response_text, metrics)
    """
    try:
        import openai
    except ImportError:
        raise ImportError("openai not installed")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = openai.OpenAI(api_key=api_key)
    
    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  # More deterministic
    )
    latency_ms = (time.time() - t0) * 1000
    
    text = response.choices[0].message.content.strip()
    
    metrics = {
        'latency_ms': latency_ms,
        'model': model,
        'tokens_used': response.usage.total_tokens if response.usage else None,
    }
    
    return text, metrics


# Provider registry
PROVIDERS = {
    'gemini': {
        'call': call_gemini,
        'models': ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro'],
        'default': 'gemini-2.0-flash',
    },
    'openai': {
        'call': call_openai,
        'models': ['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo'],
        'default': 'gpt-4o-mini',
    },
}


def get_provider_for_model(model_id: str) -> str:
    """Determine provider from model ID."""
    if model_id.startswith('gemini'):
        return 'gemini'
    elif model_id.startswith('gpt'):
        return 'openai'
    else:
        # Default to gemini
        return 'gemini'


def get_llm_completion(
    prompt: str,
    model: str = "gemini-2.0-flash",
    text_hash: str = "",
    prompt_hash: str = "",
    max_attempts: int = 3,
    timeout_s: float = 60.0,
    use_cache: bool = True,
) -> LLMResult:
    """
    Get LLM completion with retry and caching.
    
    Args:
        prompt: The prompt to send
        model: Model ID (determines provider)
        text_hash: Hash of input text (for cache key)
        prompt_hash: Hash of prompt template (for cache key)
        max_attempts: Max retry attempts
        timeout_s: Total timeout
        use_cache: Whether to use caching
        
    Returns:
        LLMResult with success/failure and response
    """
    # Check cache first
    if use_cache and text_hash and prompt_hash:
        cache_key = compute_cache_key(text_hash, prompt_hash, model)
        cached = get_cached_response(cache_key)
        if cached:
            logger.info(f"Cache hit for {cache_key[:12]}")
            return LLMResult(
                success=True,
                text=cached,
                model_id=model,
                cached=True,
                attempts=0,
            )
    else:
        cache_key = None
    
    # Get provider
    provider_name = get_provider_for_model(model)
    provider = PROVIDERS.get(provider_name)
    
    if not provider:
        return LLMResult(
            success=False,
            error_code=ErrorCode.PROVIDER_UNAVAILABLE,
            error_message=f"Unknown provider for model: {model}",
            model_id=model,
        )
    
    call_fn = provider['call']
    
    # Retry loop with exponential backoff
    start_time = time.time()
    last_error = None
    
    for attempt in range(1, max_attempts + 1):
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout_s:
            return LLMResult(
                success=False,
                error_code=ErrorCode.TIMEOUT,
                error_message=f"Timeout after {elapsed:.1f}s",
                model_id=model,
                attempts=attempt - 1,
            )
        
        try:
            text, metrics = call_fn(prompt, model)
            latency_ms = metrics.get('latency_ms', 0)
            
            # Cache successful response
            if use_cache and cache_key:
                save_to_cache(cache_key, text, model)
            
            return LLMResult(
                success=True,
                text=text,
                model_id=model,
                latency_ms=latency_ms,
                cached=False,
                attempts=attempt,
            )
            
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            # Check if rate limited
            is_rate_limit = (
                'rate' in error_str or 
                '429' in error_str or
                'quota' in error_str or
                'resource exhausted' in error_str
            )
            
            if is_rate_limit and attempt < max_attempts:
                # Exponential backoff with jitter
                base_delay = 2 ** attempt  # 2, 4, 8 seconds
                jitter = random.uniform(0, base_delay * 0.5)
                delay = min(base_delay + jitter, 30)  # Cap at 30s
                
                logger.warning(f"Rate limited, retry {attempt}/{max_attempts} in {delay:.1f}s")
                time.sleep(delay)
                continue
            
            # Other errors: don't retry
            break
    
    # All attempts failed
    error_str = str(last_error) if last_error else "Unknown error"
    
    if 'rate' in error_str.lower() or 'quota' in error_str.lower():
        error_code = ErrorCode.RATE_LIMITED
    else:
        error_code = ErrorCode.API_ERROR
    
    return LLMResult(
        success=False,
        error_code=error_code,
        error_message=error_str[:200],  # Truncate
        model_id=model,
        attempts=max_attempts,
    )


def clear_cache() -> int:
    """Clear all cached responses. Returns count of items cleared."""
    if not CACHE_DIR.exists():
        return 0
    
    count = 0
    for f in CACHE_DIR.glob("*.json"):
        f.unlink()
        count += 1
    
    return count
