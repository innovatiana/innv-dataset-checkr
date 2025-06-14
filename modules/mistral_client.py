"""
Mistral API Client Module
Handles all interactions with Mistral API for AI-powered annotation validation
"""

import os
import json
import time
import requests
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import hashlib
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MistralClient:
    """Client for interacting with Mistral API"""
    
    def __init__(self, api_key: Optional[str] = None, 
                 base_url: str = "https://api.mistral.ai/v1",
                 cache_ttl: int = 3600,
                 max_retries: int = 3,
                 rate_limit_per_minute: int = 60):
        """
        Initialize Mistral API client
        
        Args:
            api_key: Mistral API key (can also use MISTRAL_API_KEY env var)
            base_url: Base URL for Mistral API
            cache_ttl: Cache time-to-live in seconds
            max_retries: Maximum number of retry attempts
            rate_limit_per_minute: Maximum requests per minute
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key required. Set MISTRAL_API_KEY env var or pass api_key parameter")
        
        self.base_url = base_url
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        self.rate_limit_per_minute = rate_limit_per_minute
        
        # Rate limiting
        self.request_times = []
        
        # Simple in-memory cache
        self._cache = {}
        self._cache_timestamps = {}
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
    
    def _get_cache_key(self, prompt: str, params: Dict) -> str:
        """Generate cache key from prompt and parameters"""
        cache_data = json.dumps({"prompt": prompt, "params": params}, sort_keys=True)
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached entry is still valid"""
        if cache_key not in self._cache:
            return False
        
        timestamp = self._cache_timestamps.get(cache_key, 0)
        return (time.time() - timestamp) < self.cache_ttl
    
    def _rate_limit(self):
        """Implement rate limiting"""
        now = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        if len(self.request_times) >= self.rate_limit_per_minute:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        self.request_times.append(now)
    
    def _make_request(self, endpoint: str, payload: Dict, retry_count: int = 0) -> Dict:
        """Make HTTP request with retry logic"""
        self._rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:  # Rate limit
                if retry_count < self.max_retries:
                    wait_time = min(2 ** retry_count, 60)
                    logger.warning(f"Rate limited. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    return self._make_request(endpoint, payload, retry_count + 1)
            
            logger.error(f"HTTP error: {e}")
            raise
        
        except requests.exceptions.RequestException as e:
            if retry_count < self.max_retries:
                wait_time = min(2 ** retry_count, 10)
                logger.warning(f"Request failed. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                return self._make_request(endpoint, payload, retry_count + 1)
            
            logger.error(f"Request error: {e}")
            raise
    
    def validate_annotation(self, 
                          media_content: Union[str, Dict],
                          annotation: Dict,
                          context: Optional[Dict] = None,
                          model: str = "mistral-large-latest") -> Dict:
        """
        Validate annotation using Mistral AI
        
        Args:
            media_content: Text content or description of media
            annotation: Annotation to validate
            context: Additional context (taxonomy, rules, etc.)
            model: Mistral model to use
            
        Returns:
            Validation result with confidence scores and suggestions
        """
        prompt = self._build_validation_prompt(media_content, annotation, context)
        
        # Check cache
        cache_key = self._get_cache_key(prompt, {"model": model})
        if self._is_cache_valid(cache_key):
            logger.info("Returning cached result")
            return self._cache[cache_key]
        
        # Make API request
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert annotation validator. Analyze the given content and annotation, then provide detailed validation feedback in JSON format."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = self._make_request("chat/completions", payload)
            result = self._parse_validation_response(response)
            
            # Cache result
            self._cache[cache_key] = result
            self._cache_timestamps[cache_key] = time.time()
            
            return result
        
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                "is_valid": None,
                "confidence": 0.0,
                "error": str(e),
                "suggestions": []
            }
    
    def _build_validation_prompt(self, content: Union[str, Dict], 
                               annotation: Dict, 
                               context: Optional[Dict]) -> str:
        """Build validation prompt for Mistral"""
        prompt_parts = []
        
        # Content description
        if isinstance(content, str):
            prompt_parts.append(f"Content: {content}")
        else:
            prompt_parts.append(f"Content Description: {json.dumps(content, indent=2)}")
        
        # Current annotation
        prompt_parts.append(f"\nCurrent Annotation: {json.dumps(annotation, indent=2)}")
        
        # Context if provided
        if context:
            if "taxonomy" in context:
                prompt_parts.append(f"\nAvailable Labels: {context['taxonomy']}")
            if "rules" in context:
                prompt_parts.append(f"\nAnnotation Rules: {context['rules']}")
        
        # Validation request
        prompt_parts.append("""
\nPlease validate this annotation and respond with a JSON object containing:
{
    "is_valid": boolean,
    "confidence": float (0-1),
    "reasoning": "explanation of validation decision",
    "issues": ["list of specific issues found"],
    "suggestions": ["list of improvement suggestions"],
    "corrected_annotation": {corrected annotation if needed}
}
""")
        
        return "\n".join(prompt_parts)
    
    def _parse_validation_response(self, response: Dict) -> Dict:
        """Parse Mistral API response"""
        try:
            content = response["choices"][0]["message"]["content"]
            result = json.loads(content)
            
            # Ensure all expected fields exist
            return {
                "is_valid": result.get("is_valid", True),
                "confidence": result.get("confidence", 0.5),
                "reasoning": result.get("reasoning", ""),
                "issues": result.get("issues", []),
                "suggestions": result.get("suggestions", []),
                "corrected_annotation": result.get("corrected_annotation", None)
            }
        
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse response: {e}")
            return {
                "is_valid": None,
                "confidence": 0.0,
                "error": "Failed to parse AI response",
                "suggestions": []
            }
    
    def batch_validate(self, items: List[Dict], 
                      batch_size: int = 10,
                      progress_callback: Optional[callable] = None) -> List[Dict]:
        """
        Validate multiple annotations in batches
        
        Args:
            items: List of dicts with 'content', 'annotation', and optional 'context'
            batch_size: Number of items to process in parallel
            progress_callback: Function to call with progress updates
            
        Returns:
            List of validation results
        """
        results = []
        total_items = len(items)
        
        for i in range(0, total_items, batch_size):
            batch = items[i:i + batch_size]
            batch_results = []
            
            for j, item in enumerate(batch):
                result = self.validate_annotation(
                    media_content=item.get("content"),
                    annotation=item.get("annotation"),
                    context=item.get("context")
                )
                batch_results.append(result)
                
                if progress_callback:
                    progress = (i + j + 1) / total_items
                    progress_callback(progress)
            
            results.extend(batch_results)
        
        return results
    
    def suggest_labels(self, content: Union[str, Dict], 
                      taxonomy: List[str],
                      model: str = "mistral-large-latest") -> Dict:
        """
        Suggest appropriate labels for content
        
        Args:
            content: Content to label
            taxonomy: Available labels
            model: Mistral model to use
            
        Returns:
            Dict with suggested labels and confidence scores
        """
        prompt = f"""
Given the following content and label taxonomy, suggest the most appropriate labels.

Content: {content if isinstance(content, str) else json.dumps(content)}

Available Labels: {taxonomy}

Respond with a JSON object:
{{
    "primary_label": "most appropriate label",
    "confidence": float (0-1),
    "alternative_labels": ["other possible labels"],
    "reasoning": "explanation"
}}
"""
        
        cache_key = self._get_cache_key(prompt, {"model": model})
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = self._make_request("chat/completions", payload)
            result = json.loads(response["choices"][0]["message"]["content"])
            
            self._cache[cache_key] = result
            self._cache_timestamps[cache_key] = time.time()
            
            return result
        
        except Exception as e:
            logger.error(f"Label suggestion failed: {e}")
            return {
                "primary_label": None,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def clear_cache(self):
        """Clear the response cache"""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "entries": len(self._cache),
            "size_bytes": sum(len(str(v)) for v in self._cache.values()),
            "oldest_entry": min(self._cache_timestamps.values()) if self._cache_timestamps else None,
            "newest_entry": max(self._cache_timestamps.values()) if self._cache_timestamps else None
        }


# Convenience function for single-use validation
def validate_with_mistral(content: str, annotation: Dict, 
                         api_key: Optional[str] = None) -> Dict:
    """
    Quick validation function for single annotations
    
    Args:
        content: Content to validate
        annotation: Annotation to check
        api_key: Mistral API key (optional if env var set)
        
    Returns:
        Validation result
    """
    client = MistralClient(api_key=api_key)
    return client.validate_annotation(content, annotation)


if __name__ == "__main__":
    # Example usage
    client = MistralClient()
    
    # Example validation
    result = client.validate_annotation(
        media_content="A large brown dog playing in the park",
        annotation={"label": "cat", "confidence": 0.9},
        context={"taxonomy": ["dog", "cat", "bird", "other"]}
    )
    
    print(json.dumps(result, indent=2))