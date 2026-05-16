import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Type
from urllib.parse import urlparse

import httpx
import requests
from dotenv import find_dotenv, load_dotenv
from loguru import logger


INVALID_PROXY_HOSTS = {"127.0.0.1", "localhost", "::1"}
INVALID_PROXY_PORT = 9
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    model: str = DEFAULT_DEEPSEEK_MODEL
    temperature: float = 0.7
    max_tokens: int = 1000


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_project_dotenv() -> Optional[str]:
    """Load only the workspace-root .env, tolerating UTF-8/UTF-16 files."""
    root_env = project_root() / ".env"
    env_path = str(root_env) if root_env.exists() else find_dotenv(usecwd=True)
    if not env_path:
        return None

    try:
        load_dotenv(env_path, encoding="utf-8", override=False)
    except UnicodeDecodeError:
        load_dotenv(env_path, encoding="utf-16", override=False)
    return env_path


def _first_env(*keys: str) -> Optional[str]:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return None


def _api_section(config: Optional[Dict[str, Any]], name: str) -> Dict[str, Any]:
    if not config:
        return {}
    section = config.get("api", {}).get(name, {})
    return section if isinstance(section, dict) else {}


def load_deepseek_config(
    config: Optional[Dict[str, Any]] = None,
    *,
    api_key: Optional[str] = None,
) -> DeepSeekConfig:
    """Load the project DeepSeek config, keeping OPENAI_* as legacy aliases."""
    load_project_dotenv()
    deepseek = _api_section(config, "deepseek")
    legacy_openai = _api_section(config, "openai")

    selected_api_key = (
        api_key
        or _first_env("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY")
        or deepseek.get("api_key")
        or legacy_openai.get("api_key")
        or ""
    )
    base_url = (
        _first_env("DEEPSEEK_BASE_URL", "OPENAI_BASE_URL")
        or deepseek.get("base_url")
        or DEFAULT_DEEPSEEK_BASE_URL
    )
    model = (
        _first_env("DEEPSEEK_MODEL", "OPENAI_MODEL")
        or deepseek.get("model")
        or DEFAULT_DEEPSEEK_MODEL
    )
    temperature = deepseek.get("temperature", legacy_openai.get("temperature", 0.7))
    max_tokens = deepseek.get("max_tokens", legacy_openai.get("max_tokens", 1000))

    return DeepSeekConfig(
        api_key=selected_api_key,
        base_url=normalize_base_url(base_url) or DEFAULT_DEEPSEEK_BASE_URL,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def normalize_base_url(base_url: Optional[str]) -> Optional[str]:
    if not base_url:
        return None
    stripped = base_url.strip()
    return stripped.rstrip("/") if stripped else None


def chat_completions_url(base_url: Optional[str]) -> str:
    normalized = normalize_base_url(base_url) or DEFAULT_DEEPSEEK_BASE_URL
    return f"{normalized}/chat/completions"


def is_invalid_blocking_proxy(proxy_url: Optional[str]) -> bool:
    if not proxy_url:
        return False

    parsed = urlparse(proxy_url)
    host = parsed.hostname
    try:
        port = parsed.port
    except ValueError:
        return False

    return host in INVALID_PROXY_HOSTS and port == INVALID_PROXY_PORT


def find_invalid_llm_proxy(environ: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = environ or os.environ
    return {
        key: value
        for key in PROXY_ENV_KEYS
        if (value := env.get(key)) and is_invalid_blocking_proxy(value)
    }


def should_disable_env_proxy() -> bool:
    invalid = find_invalid_llm_proxy()
    if invalid:
        names = ", ".join(sorted(invalid))
        logger.warning(
            "Ignoring invalid local proxy for LLM requests: {}. "
            "127.0.0.1:9 is a blocking placeholder, not a real proxy.",
            names,
        )
        return True
    return False


def openai_client_kwargs(
    api_key: str,
    base_url: Optional[str] = None,
    *,
    async_client: bool = False,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"api_key": api_key}
    normalized_base_url = normalize_base_url(base_url)
    if normalized_base_url:
        kwargs["base_url"] = normalized_base_url

    if should_disable_env_proxy():
        kwargs["http_client"] = (
            httpx.AsyncClient(trust_env=False)
            if async_client
            else httpx.Client(trust_env=False)
        )

    return kwargs


def create_openai_client(
    client_cls: Type[Any],
    api_key: str,
    base_url: Optional[str] = None,
) -> Any:
    return client_cls(**openai_client_kwargs(api_key, base_url, async_client=False))


def create_async_openai_client(
    client_cls: Type[Any],
    api_key: str,
    base_url: Optional[str] = None,
) -> Any:
    return client_cls(**openai_client_kwargs(api_key, base_url, async_client=True))


def llm_requests_post(url: str, **kwargs: Any) -> requests.Response:
    """POST to an LLM endpoint, bypassing only the known invalid proxy."""
    if should_disable_env_proxy():
        with requests.Session() as session:
            session.trust_env = False
            return session.post(url, **kwargs)
    return requests.post(url, **kwargs)
