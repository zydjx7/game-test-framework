#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Update the root .env for DeepSeek API usage."""

import argparse
import getpass
import os
import sys
from pathlib import Path

from loguru import logger
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.client_helpers import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    create_openai_client,
    load_deepseek_config,
)


LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add(str(LOG_DIR / "api_update.log"), level="DEBUG", rotation="1 MB")

ENV_PATH = PROJECT_ROOT / ".env"


def read_env_file() -> list[str]:
    if not ENV_PATH.exists():
        return []
    try:
        return ENV_PATH.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return ENV_PATH.read_text(encoding="utf-16").splitlines()


def write_env_file(lines: list[str]) -> None:
    ENV_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def upsert_env(updates: dict[str, str], remove_keys: set[str] | None = None) -> None:
    remove_keys = remove_keys or set()
    lines = read_env_file()
    seen: set[str] = set()
    output: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            output.append(line)
            continue

        key, _ = line.split("=", 1)
        key = key.strip()
        if key in remove_keys:
            continue
        if key in updates:
            output.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            output.append(line)

    for key, value in updates.items():
        if key not in seen:
            output.append(f"{key}={value}")

    write_env_file(output)


def update_deepseek_config(api_key: str | None, model: str | None) -> None:
    updates = {
        "DEEPSEEK_BASE_URL": DEFAULT_DEEPSEEK_BASE_URL,
        "DEEPSEEK_MODEL": model or DEFAULT_DEEPSEEK_MODEL,
    }
    if api_key:
        updates["DEEPSEEK_API_KEY"] = api_key

    remove_keys = {"API_TYPE", "OPENAI_BASE_URL", "OPENAI_MODEL"}
    if api_key:
        remove_keys.add("OPENAI_API_KEY")

    upsert_env(updates, remove_keys=remove_keys)
    logger.info("Updated root .env for DeepSeek at {}", ENV_PATH)
    logger.info("Model: {}", updates["DEEPSEEK_MODEL"])
    logger.info("Base URL: {}", updates["DEEPSEEK_BASE_URL"])


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
        logger.info("DeepSeek API key test passed. Model: {}", config.model)
        logger.info("Received response: {}", response.choices[0].message.content)
        return True
    except Exception as exc:
        logger.error("DeepSeek API key test failed: {}", exc)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Update DeepSeek API settings in the root .env")
    parser.add_argument("--key", help="New DeepSeek API key. If omitted with --prompt-key, input is hidden.")
    parser.add_argument("--prompt-key", action="store_true", help="Prompt for the DeepSeek API key without echo.")
    parser.add_argument("--model", default=DEFAULT_DEEPSEEK_MODEL, help="DeepSeek model name.")
    parser.add_argument("--test", action="store_true", help="Test the configured DeepSeek API key.")
    args = parser.parse_args()

    api_key = args.key
    if args.prompt_key:
        api_key = getpass.getpass("DeepSeek API key: ").strip()

    if api_key or args.model:
        update_deepseek_config(api_key, args.model)

    if args.test or not (api_key or args.model):
        return 0 if test_api_key() else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
