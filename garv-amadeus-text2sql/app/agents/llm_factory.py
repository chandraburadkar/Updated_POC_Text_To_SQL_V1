# app/agents/llm_factory.py
from __future__ import annotations

import os
import logging
from typing import Optional, Any, List

from dotenv import load_dotenv
from pydantic import Field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger(__name__)

# Load .env automatically whenever this module is imported
load_dotenv(override=True)


# ============================================================
# Ollama HTTP Chat Model (LangChain-compatible)
# ============================================================
class ChatOllamaHTTP(BaseChatModel):
    """
    Minimal LangChain-compatible chat model using Ollama HTTP API.

    Why we do this:
    - langchain-ollama can force newer langchain-core versions (1.x) which
      conflicts with your current langchain/langgraph family (0.3.x).
    - This custom wrapper uses plain HTTP, so we keep dependencies stable.

    Compatible with:
    - langchain-core 0.3.x
    """

    model: str = Field(...)
    base_url: str = Field(default="http://localhost:11434")
    temperature: float = Field(default=0.0)

    @property
    def _llm_type(self) -> str:
        return "ollama_http"

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """
        Convert chat messages to a simple prompt format.

        Example:
        SYSTEM: ...
        HUMAN: ...
        AI: ...
        """
        parts = []
        for m in messages:
            role = m.type  # "system" | "human" | "ai"
            parts.append(f"{role.upper()}: {m.content}")
        return "\n".join(parts) + "\nAI:"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Core LangChain hook that returns ChatResult.
        """
        import requests  # keep import local to avoid issues if not installed

        prompt = self._convert_messages_to_prompt(messages)

        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": float(self.temperature)},
        }

        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(
                f"Ollama HTTP call failed. "
                f"Check Ollama is running at {self.base_url} and model '{self.model}' exists. "
                f"Original error: {e}"
            )

        text = data.get("response", "")

        if stop:
            for s in stop:
                text = text.split(s)[0]

        gen = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[gen])


# ============================================================
# Gemini Chat Model Factory
# ============================================================
def _get_gemini_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Returns Gemini chat model using langchain-google-genai.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception as e:
        raise ImportError(
            "langchain-google-genai is not installed or incompatible. "
            "Install it with: pip install langchain-google-genai"
        ) from e

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment. Set it in .env or export it.")

    # Use a model that is available in your `client.models.list()` output
    # You listed: models/gemini-2.5-flash, models/gemini-2.0-flash-001, etc.
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")

    # langchain-google-genai expects model names without "models/" prefix.
    # You must pass "gemini-2.0-flash-001" not "models/gemini-2.0-flash-001".
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
    )
    return llm


# ============================================================
# Ollama Chat Model Factory
# ============================================================
def _get_ollama_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Returns Ollama local model via HTTP wrapper.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

    return ChatOllamaHTTP(
        model=model,
        base_url=base_url,
        temperature=temperature,
    )


# ============================================================
# Public function used by agents
# ============================================================
def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Unified LLM selector.
    Controlled by env var:

    LLM_PROVIDER=ollama  -> local Ollama
    LLM_PROVIDER=gemini  -> Gemini API

    Default: ollama (safe to avoid quota issues)
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

    if provider == "gemini":
        logger.info("Using Gemini LLM provider")
        return _get_gemini_llm(temperature=temperature)

    logger.info("Using Ollama LLM provider")
    return _get_ollama_llm(temperature=temperature)