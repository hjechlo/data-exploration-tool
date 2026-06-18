"""
LLM Engine Module
--------------------
Handles heterogeneous routing between native Azure OpenAI endpoints and standard OpenAI-compatible endpoints
"""

from typing import Optional, Union
from openai import AzureOpenAI, OpenAI


class AzureLLMEngine:
    def __init__(self, api_key: str, api_version: str = "2024-05-01-preview"):
        """
        Used to insert the API key and version once, then route it to the relevant client wrapper
        """
        if not api_key:
            raise ValueError("CRITICAL: API Key must be explicitly provided.")
        self.api_key = api_key
        self.api_version = api_version
        self._client_cache = {}

    def _get_client(self, endpoint: str, is_native_azure: bool) -> Union[AzureOpenAI, OpenAI]:
        """Dynamically builds or retrieves the exact required SDK client."""
        cache_key = (endpoint, is_native_azure)
        if cache_key not in self._client_cache:
            if is_native_azure:
                # Used for native Azure OpenAI deployments (AzureOpenAI client)
                self._client_cache[cache_key] = AzureOpenAI(
                    azure_endpoint=endpoint,
                    api_key=self.api_key,
                    api_version=self.api_version
                )
            else:
                # Used for Kimi, GPT-OSS, and Maverick (Standard OpenAI wrapper targeting Azure base_url)
                self._client_cache[cache_key] = OpenAI(
                    base_url=endpoint,
                    api_key=self.api_key
                )
        return self._client_cache[cache_key]

    def generate_response(
        self, 
        endpoint: str,
        deployment_name: str, 
        system_prompt: str, 
        user_payload: str, 
        is_native_azure: bool = False,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Routes the inference payload through the correct client wrapper.
        """
        if not endpoint:
            raise ValueError("CRITICAL: Target endpoint string is missing or empty.")
            
        try:
            client = self._get_client(endpoint, is_native_azure)
            
            # 1. Dynamically build parameters to avoid sending explicit 'None' values to Azure
            kwargs = {
                "model": deployment_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload}
                ],
                "temperature": temperature
            }
            
            # Only inject max_tokens if it's explicitly set to an integer value
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            
            # 2. Dispatch payload package safely
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(
                f"API execution failure on model [{deployment_name}]: {str(e)}"
            ) from e