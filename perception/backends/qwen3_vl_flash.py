"""Qwen3-VL-Flash backend via DashScope's OpenAI-compatible endpoint.

We deliberately reuse the project's existing ``openai`` SDK rather than
installing the ``dashscope`` SDK, mirroring how DeepSeek is wired
(see CLAUDE.md). DashScope exposes an OpenAI-compatible API, so the only
differences from the DeepSeek path are the base_url, the model name, and
sending an image part in the message content.

Config (Phase 1 design doc §8):
- base_url (Beijing): https://dashscope.aliyuncs.com/compatible-mode/v1
- model: qwen3-vl-flash
- API key: DASHSCOPE_API_KEY in the project .env
- image: base64 data URI (data:image/png;base64,...), OpenAI vision format
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from openai import OpenAI

from src.llm.client_helpers import load_project_dotenv, openai_client_kwargs


@dataclass
class BackendResponse:
    """Uniform return type every VLM backend produces."""

    text: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Qwen3VLFlashBackend:
    DEFAULT_MODEL = "qwen3-vl-flash"
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        load_project_dotenv()
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or ""
        if not self.api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY is not set. Add `DASHSCOPE_API_KEY=sk-...` to "
                "the project .env (阿里云百炼 key)."
            )
        self.model = model or os.getenv("QWEN_VL_MODEL") or self.DEFAULT_MODEL
        self.base_url = (
            base_url or os.getenv("DASHSCOPE_BASE_URL") or self.DEFAULT_BASE_URL
        )
        self.temperature = temperature
        self.client = OpenAI(**openai_client_kwargs(self.api_key, self.base_url))

    def infer(self, prompt: str, image_data_uri: str) -> BackendResponse:
        """Send one prompt + one image, return text + latency + token usage."""

        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                    ],
                }
            ],
            temperature=self.temperature,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        return BackendResponse(
            text=text,
            latency_ms=latency_ms,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )
