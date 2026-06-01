"""Unit tests for ``perception.vlm_perceptor`` (no real API calls).

A FakeBackend returns canned text so we can test JSON parsing, the
no-raise contract, and metadata propagation without hitting DashScope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from perception import GameState, VLMPerceptor, encode_image_data_uri, parse_vlm_json


@dataclass
class FakeResponse:
    text: str
    latency_ms: float = 12.3
    prompt_tokens: int = 100
    completion_tokens: int = 10


class FakeBackend:
    def __init__(self, text: str = '{"ammo_level": "low", "ammo": 7}', raise_exc=None):
        self.text = text
        self.raise_exc = raise_exc
        self.calls = 0

    def infer(self, prompt: str, image_data_uri: str) -> FakeResponse:
        self.calls += 1
        if self.raise_exc is not None:
            raise self.raise_exc
        return FakeResponse(text=self.text)


@pytest.fixture
def screen() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


class TestParseVlmJson:
    def test_plain_json(self):
        parsed, err = parse_vlm_json('{"ammo_level": "high", "ammo": 26}')
        assert err is None
        assert parsed == {"ammo_level": "high", "ammo": 26}

    def test_markdown_fenced_json(self):
        parsed, err = parse_vlm_json('```json\n{"ammo_level": "low", "ammo": 5}\n```')
        assert err is None
        assert parsed["ammo"] == 5

    def test_json_with_surrounding_prose(self):
        parsed, err = parse_vlm_json('The ammo is low. {"ammo_level": "low", "ammo": 3} done')
        assert err is None
        assert parsed["ammo"] == 3

    def test_empty_is_error(self):
        parsed, err = parse_vlm_json("")
        assert parsed is None
        assert err == "empty response"

    def test_no_json_is_error(self):
        parsed, err = parse_vlm_json("I cannot see any ammo counter.")
        assert parsed is None
        assert err == "no JSON object found"


class TestEncodeImageDataUri:
    def test_produces_png_data_uri(self, screen):
        uri = encode_image_data_uri(screen)
        assert uri.startswith("data:image/png;base64,")
        assert len(uri) > len("data:image/png;base64,")


class TestVLMPerceptor:
    def test_parses_ammo_and_stashes_level(self, screen):
        perceptor = VLMPerceptor(FakeBackend('{"ammo_level": "low", "ammo": 7}'))
        state = perceptor.perceive(screen)

        assert isinstance(state, GameState)
        assert state.ammo == 7
        assert state.raw_response["vlm_level"] == "low"
        assert state.raw_response["latency_ms"] == 12.3
        assert state.raw_response["prompt_tokens"] == 100

    def test_null_ammo_yields_none(self, screen):
        perceptor = VLMPerceptor(FakeBackend('{"ammo_level": "unknown", "ammo": null}'))
        state = perceptor.perceive(screen)
        assert state.ammo is None
        assert state.raw_response["vlm_level"] == "unknown"

    def test_malformed_json_does_not_raise(self, screen):
        perceptor = VLMPerceptor(FakeBackend("not json at all"))
        state = perceptor.perceive(screen)
        assert state.ammo is None
        assert "error" in state.raw_response
        assert state.raw_response["raw_text"] == "not json at all"

    def test_backend_exception_does_not_raise(self, screen):
        perceptor = VLMPerceptor(FakeBackend(raise_exc=RuntimeError("network down")))
        state = perceptor.perceive(screen)
        assert state.ammo is None
        assert "backend" in state.raw_response["error"]
