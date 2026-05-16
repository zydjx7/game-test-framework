#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Validate the configured DeepSeek API key without printing it."""

import os
import sys

from loguru import logger
from openai import OpenAI

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.llm.client_helpers import create_openai_client, load_deepseek_config


LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "api_test.log"), level="DEBUG", rotation="1 MB")


def test_api_key() -> bool:
    config = load_deepseek_config()
    if not config.api_key:
        logger.error("No API key found. Set DEEPSEEK_API_KEY in the root .env.")
        return False

    client = create_openai_client(
        OpenAI,
        api_key=config.api_key,
        base_url=config.base_url,
    )

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5,
        )
        logger.info("DeepSeek API key is valid. Model: {}", config.model)
        logger.info("Received response: {}", response.choices[0].message.content)
        return True
    except Exception as exc:
        logger.error("DeepSeek API key validation failed: {}", exc)
        return False


if __name__ == "__main__":
    success = test_api_key()
    print("DeepSeek API key validation passed" if success else "DeepSeek API key validation failed")
    raise SystemExit(0 if success else 1)
