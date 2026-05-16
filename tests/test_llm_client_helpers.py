import yaml

from src.llm import client_helpers
from src.llm.gpt_client import GPTClient


PROXY_KEYS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]


def clear_proxy_env(monkeypatch):
    for key in PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_invalid_proxy_detection(monkeypatch):
    clear_proxy_env(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")

    invalid = client_helpers.find_invalid_llm_proxy()

    assert invalid
    assert all(value == "http://127.0.0.1:9" for value in invalid.values())
    assert any(key.lower() == "https_proxy" for key in invalid)


def test_valid_proxy_is_preserved(monkeypatch):
    clear_proxy_env(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")

    kwargs = client_helpers.openai_client_kwargs("test-key", "https://api.deepseek.com")

    assert kwargs["base_url"] == "https://api.deepseek.com"
    assert "http_client" not in kwargs


def test_deepseek_config_prefers_deepseek_env(monkeypatch):
    clear_proxy_env(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-v4-pro")

    config = client_helpers.load_deepseek_config()

    assert config.api_key == "deepseek-key"
    assert config.base_url == "https://api.deepseek.com"
    assert config.model == "deepseek-v4-flash"


def test_deepseek_config_supports_legacy_openai_env(monkeypatch):
    clear_proxy_env(monkeypatch)
    monkeypatch.setattr(client_helpers, "load_project_dotenv", lambda: None)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-v4-pro")

    config = client_helpers.load_deepseek_config()

    assert config.api_key == "legacy-key"
    assert config.base_url == "https://api.deepseek.com"
    assert config.model == "deepseek-v4-pro"


def test_invalid_proxy_disables_openai_sdk_env_proxy(monkeypatch):
    clear_proxy_env(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://localhost:9")

    kwargs = client_helpers.openai_client_kwargs("test-key")

    assert "http_client" in kwargs
    assert kwargs["http_client"].trust_env is False
    kwargs["http_client"].close()


def test_gpt_client_passes_deepseek_base_url(monkeypatch, tmp_path):
    clear_proxy_env(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "api": {
                    "deepseek": {
                        "api_key": "",
                        "base_url": "https://api.deepseek.com",
                        "model": "deepseek-v4-flash",
                        "temperature": 0.7,
                        "max_tokens": 1000,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    class FakeAsyncOpenAI:
        kwargs = None

        def __init__(self, **kwargs):
            FakeAsyncOpenAI.kwargs = kwargs

    monkeypatch.setattr("src.llm.gpt_client.AsyncOpenAI", FakeAsyncOpenAI)

    GPTClient(config_path=str(config_path))

    assert FakeAsyncOpenAI.kwargs["api_key"] == "test-key"
    assert FakeAsyncOpenAI.kwargs["base_url"] == "https://api.deepseek.com"
    assert "http_client" not in FakeAsyncOpenAI.kwargs
