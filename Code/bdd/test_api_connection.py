#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Smoke-test the configured DeepSeek API connection."""

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

logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add(os.path.join(LOG_DIR, "api_test.log"), level="DEBUG", rotation="1 MB")


def test_connection() -> bool:
    config = load_deepseek_config()
    logger.info("Testing DeepSeek API connection")
    logger.info("Model: {}", config.model)
    logger.info("Base URL: {}", config.base_url)

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
            messages=[
                {"role": "system", "content": "You are a concise game testing assistant."},
                {"role": "user", "content": "Return exactly: API connection OK"},
            ],
            max_tokens=20,
        )
        logger.info("DeepSeek API call succeeded")
        logger.info("Response: {}", response.choices[0].message.content)
        return True
    except Exception as exc:
        logger.error("DeepSeek API call failed: {}", exc)
        return False


if __name__ == "__main__":
    success = test_connection()
    print("DeepSeek API connection test passed" if success else "DeepSeek API connection test failed")
    raise SystemExit(0 if success else 1)
