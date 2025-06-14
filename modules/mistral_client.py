"""
Mistral API Client Module
Handles all interactions with Mistral API for AI-powered annotation validation
"""

import os
import json
import time
import requests
from typing import Dict, List, Optional, Any, Union
import hashlib
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
        self.api_key = api_key or "auBrR2Mdesaz2XeoW0VrSdi3ClKthQ0Z"
        if not self.api_key:
            raise ValueError("Mistral API key required. Set MISTRAL_API_KEY env var or pass api_key parameter")

        self.base_url = base_url
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        self.rate_limit_per_minute = rate_limit_per_minute

        self.request_times = []
        self._cache = {}
        self._cache_timestamps = {}

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    def _get_cache_key(self, prompt: str, params: Dict) -> str:
        cache_data = json.dumps({"prompt": prompt, "params": params}, sort_keys=True)
        return hashlib.md5(cache_data.encode()).hexdigest()

    def _is_cache_valid(self, cache_key: str) -> bool:
        if cache_key not in self._cache:
            return False
        timestamp = self._cache_timestamps.get(cache_key, 0)
        return (time.time() - timestamp) < self.cache_ttl

    def _rate_limit(self):
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 60]
        if len(self.request_times) >= self.rate_limit_per_minute:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        self.request_times.append(now)

    def _make_request(self, endpoint: str, payload: Dict, retry_count: int = 0) -> Dict:
        self._rate_limit()
        url = f"{self.base_url}/{endpoint}"
        try:
            response = self.session.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and retry_count < self.max_retries:
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
        prompt = self._build_validation_prompt(media_content, annotation, context)
        cache_key = self._get_cache_key(prompt, {"model": model})
        if self._is_cache_valid(cache_key):
            logger.info("Returning cached result")
            return self._cache[cache_key]

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
            "temperature": 0.1
        }

        try:
            response = self._make_request("chat/completions", payload)
            result = self._parse_validation_response(response)
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
                checks = context.get("selected_checks", []) if context else []
            
                prompt_parts = []
                prompt_parts.append(f"Content: {content if isinstance(content, str) else json.dumps(content, indent=2)}")
                prompt_parts.append(f"\nAnnotation: {json.dumps(annotation, indent=2)}")
                prompt_parts.append("\nPerform the following validation checks:")
            
                if "Check label correctness" in checks:
                    prompt_parts.append("- Verify if the assigned label is correct.")
                if "Check content consistency" in checks:
                    prompt_parts.append("- Analyze whether the content is logical and coherent.")
                if "Detect bias" in checks:
                    prompt_parts.append("- Detect bias in the annotation or language.")
                if "Check annotation completeness" in checks:
                    prompt_parts.append("- Check if the annotation fully covers the relevant information.")
                if "Suggest corrections" in checks:
                    prompt_parts.append("- Suggest improved or corrected annotations, if needed.")
            
                prompt_parts.append("""
            Return your response in JSON format:
            {
              "is_valid": true/false,
              "confidence": float between 0 and 1,
              "reasoning": "Explain the assessment",
              "issues": ["List of detected issues"],
              "suggestions": ["List of proposed corrections"],
              "corrected_annotation": {... if applicable },
              "bias_flagged": true/false,
              "bias_notes": "Explanation of any identified bias"
            }
            """)
                return "\n".join(prompt_parts)

    def _parse_validation_response(self, response: Dict) -> Dict:
        try:
            content = response["choices"][0]["message"]["content"]
            result = json.loads(content)
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

    def suggest_labels(self, content: Union[str, Dict], taxonomy: List[str], model: str = "mistral-large-latest") -> Dict:
        prompt = f"""
Given the following content and label taxonomy, suggest the most appropriate labels.

Content: {content if isinstance(content, str) else json.dumps(content)}
Available Labels: {taxonomy}

Respond with a JSON object like:
{{
    "primary_label": "label",
    "confidence": float (0-1),
    "alternative_labels": ["list"],
    "reasoning": "why you picked it"
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
            "temperature": 0.1
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

    def batch_validate(self, items: List[Dict], batch_size: int = 10, progress_callback: Optional[callable] = None) -> List[Dict]:
        results = []
        total = len(items)
        for i in range(0, total, batch_size):
            batch = items[i:i + batch_size]
            for j, item in enumerate(batch):
                result = self.validate_annotation(
                    media_content=item.get("content"),
                    annotation=item.get("annotation"),
                    context=item.get("context")
                )
                results.append(result)
                if progress_callback:
                    progress_callback((i + j + 1) / total)
        return results

    def clear_cache(self):
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Cache cleared")

    def get_cache_stats(self) -> Dict:
        return {
            "entries": len(self._cache),
            "size_bytes": sum(len(str(v)) for v in self._cache.values()),
            "oldest_entry": min(self._cache_timestamps.values()) if self._cache_timestamps else None,
            "newest_entry": max(self._cache_timestamps.values()) if self._cache_timestamps else None
        }


def validate_with_mistral(content: str, annotation: Dict, api_key: Optional[str] = None) -> Dict:
    client = MistralClient(api_key=api_key)
    return client.validate_annotation(content, annotation)


if __name__ == "__main__":
    # Quick test (set your API key in env or pass directly)
    client = MistralClient()
    result = client.validate_annotation(
        media_content="A large brown dog playing in the park",
        annotation={"label": "cat", "confidence": 0.9},
        context={"taxonomy": ["dog", "cat", "bird", "other"]}
    )
    print(json.dumps(result, indent=2))
