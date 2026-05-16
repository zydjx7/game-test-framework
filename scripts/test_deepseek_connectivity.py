"""DeepSeek API connectivity smoke test.

Sends a 5-token chat completion to verify:
- .env is found and parsed
- API key is present and valid (not 401)
- base_url + model identifier are accepted
- proxy bypass logic in client_helpers actually works

Cost: ~1 input + 2-3 output tokens (~$0.0000003 per run; negligible).

Usage::

    python scripts/test_deepseek_connectivity.py

Exit code 0 on success, 1 on any failure. Failure modes printed so you
can act: missing key, wrong model name, network error, proxy interference,
etc.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.llm.client_helpers import (  # noqa: E402
    chat_completions_url,
    find_invalid_llm_proxy,
    llm_requests_post,
    load_deepseek_config,
    load_project_dotenv,
)


def _mask(secret: str) -> str:
    if not secret:
        return "(empty)"
    if len(secret) <= 12:
        return secret[:4] + "***"
    return f"{secret[:8]}...{secret[-4:]}"


def main() -> int:
    print("=== DeepSeek API connectivity test ===\n")

    env_path = load_project_dotenv()
    print(f"[1/4] .env discovered at: {env_path or '(none -- relying on shell env)'}")

    cfg = load_deepseek_config()
    print(f"[2/4] config loaded:")
    print(f"      base_url = {cfg.base_url}")
    print(f"      model    = {cfg.model}")
    print(f"      api_key  = {_mask(cfg.api_key)}")

    if not cfg.api_key:
        print("\nFAIL: no API key found. Expected one of these env vars to be set:")
        print("      DEEPSEEK_API_KEY  |  OPENAI_API_KEY  |  LLM_API_KEY")
        return 1

    invalid_proxies = find_invalid_llm_proxy()
    if invalid_proxies:
        print(f"      detected invalid local proxies: {sorted(invalid_proxies)}")
        print("      client_helpers will bypass these automatically")

    url = chat_completions_url(cfg.base_url)
    print(f"\n[3/4] POST {url}")

    payload = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": "Reply with exactly one word: pong"}],
        "max_tokens": 5,
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = llm_requests_post(url, json=payload, headers=headers, timeout=30)
    except Exception as exc:
        print(f"\nFAIL: request raised {type(exc).__name__}: {exc}")
        print("      possible causes: DNS / firewall / VPN / network down")
        return 1

    print(f"      HTTP {resp.status_code}")

    if resp.status_code == 401:
        print("\nFAIL: 401 Unauthorized -- API key invalid or revoked.")
        print("      Re-issue the key at https://platform.deepseek.com/api_keys")
        return 1
    if resp.status_code == 402:
        print("\nFAIL: 402 -- DeepSeek account out of credit. Top up at the console.")
        return 1
    if resp.status_code == 404:
        print(f"\nFAIL: 404 -- model '{cfg.model}' not found.")
        print("      DeepSeek currently has: deepseek-v4-flash, deepseek-v4-pro,")
        print("      deepseek-chat (deprecated 2026-07-24), deepseek-reasoner (deprecated).")
        return 1
    if resp.status_code != 200:
        print(f"\nFAIL: HTTP {resp.status_code}")
        print(f"      body (first 500 chars): {resp.text[:500]}")
        return 1

    try:
        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
    except (KeyError, ValueError) as exc:
        print(f"\nFAIL parsing response: {exc!r}")
        print(f"      body (first 500 chars): {resp.text[:500]}")
        return 1

    print(f"\n[4/4] reply: {reply!r}")
    print(f"      tokens: prompt={usage.get('prompt_tokens')}, "
          f"completion={usage.get('completion_tokens')}, "
          f"total={usage.get('total_tokens')}")
    print("\n=== DEEPSEEK API OK ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
