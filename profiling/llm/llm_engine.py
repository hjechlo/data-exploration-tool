"""LLM Engine — routes inference between Azure OpenAI and OpenAI-compatible endpoints."""

from typing import Optional, Union

from openai import AzureOpenAI, OpenAI


class AzureLLMEngine:
    """Unified client wrapper for native Azure OpenAI and OpenAI-compatible endpoints.

    Builds and caches the appropriate SDK client per (endpoint, is_native_azure)
    combination so callers never have to manage client lifetimes.
    """

    def __init__(self, api_key: str, api_version: str = "2024-05-01-preview") -> None:
        if not api_key:
            raise ValueError("CRITICAL: API key must be explicitly provided.")
        self.api_key = api_key
        self.api_version = api_version
        self._client_cache: dict[tuple, Union[AzureOpenAI, OpenAI]] = {}

    def _get_client(self, endpoint: str, is_native_azure: bool) -> Union[AzureOpenAI, OpenAI]:
        """Return a cached (or newly built) SDK client for the given endpoint."""
        key = (endpoint, is_native_azure)
        if key not in self._client_cache:
            if is_native_azure:
                self._client_cache[key] = AzureOpenAI(
                    azure_endpoint=endpoint,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
            else:
                # OpenAI-compatible wrapper for Kimi, GPT-OSS, Maverick, etc.
                self._client_cache[key] = OpenAI(
                    base_url=endpoint,
                    api_key=self.api_key,
                )
        return self._client_cache[key]

    def generate_response(
        self,
        endpoint: str,
        deployment_name: str,
        system_prompt: str,
        user_payload: str,
        is_native_azure: bool = False,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Route the inference payload through the correct client wrapper."""
        if not endpoint:
            raise ValueError("CRITICAL: Target endpoint string is missing or empty.")

        client = self._get_client(endpoint, is_native_azure)
        kwargs: dict = {
            "model": deployment_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            "temperature": temperature,
        }
        # Omit max_tokens entirely rather than passing None, which some
        # Azure deployments reject.
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as exc:
            raise RuntimeError(
                f"API execution failure on model [{deployment_name}]: {exc}"
            ) from exc