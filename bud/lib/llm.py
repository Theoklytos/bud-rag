"""LLM client for Bud RAG Pipeline."""

import requests

from bud.lib.errors import LLMError, LLMTimeoutError


class LLMClient:
    """Client for interacting with LLM providers."""

    def __init__(self, config: dict):
        self._cfg = config["llm"]

    def complete(self, system: str, user: str) -> str:
        """Generate a completion from the LLM.

        Args:
            system: System prompt
            user: User message

        Returns:
            Generated response text

        Raises:
            LLMError: If the API call fails
        """
        provider = self._cfg["provider"]
        if provider == "ollama":
            return self._ollama(system, user)
        elif provider == "claude":
            return self._claude(system, user)
        elif provider == "grok":
            return self._openai_compat(system, user)
        else:
            raise LLMError(f"Unknown provider: {provider}")

    def _ollama(self, system: str, user: str) -> str:
        """Call Ollama API."""
        url = f"{self._cfg['base_url'].rstrip('/')}/api/chat"
        payload = {
            "model": self._cfg["model"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        timeout = self._cfg.get("timeout_seconds", 60)
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
        except requests.exceptions.Timeout:
            raise LLMTimeoutError("Ollama request timed out")
        except requests.exceptions.RequestException as e:
            raise LLMError(f"Ollama connection error: {e}")
        if resp.status_code != 200:
            raise LLMError(f"Ollama returned {resp.status_code}: {resp.text}")
        return resp.json()["message"]["content"]

    def _openai_compat(self, system: str, user: str) -> str:
        """Call OpenAI-compatible API (e.g., Grok)."""
        url = f"{self._cfg['base_url'].rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._cfg['api_key']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._cfg["model"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        timeout = self._cfg.get("timeout_seconds", 60)
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except requests.exceptions.Timeout:
            raise LLMTimeoutError("Request timed out")
        except requests.exceptions.RequestException as e:
            raise LLMError(f"Connection error: {e}")
        if resp.status_code != 200:
            raise LLMError(f"API returned {resp.status_code}: {resp.text}")
        return resp.json()["choices"][0]["message"]["content"]

    def _claude(self, system: str, user: str) -> str:
        """Call Anthropic Claude API."""
        try:
            import anthropic
        except ImportError:
            raise LLMError("anthropic package not installed")
        client = anthropic.Anthropic(api_key=self._cfg["api_key"])
        try:
            msg = client.messages.create(
                model=self._cfg["model"],
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as e:
            raise LLMError(f"Claude API error: {e}")
        return msg.content[0].text
