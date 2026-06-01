"""VLM-based perceptor: screenshot -> VLM -> parsed GameState.

Backend-agnostic. It owns the prompt, the image encoding, and the JSON
parsing; the actual model call is delegated to a backend object exposing
``infer(prompt, image_data_uri) -> BackendResponse`` (see
``perception/backends/``). This keeps the multi-backend Phase 4 work to
"write another backend", not "rewrite the perceptor".

Honours the GameStatePerceptor no-raise contract: any failure (encode,
network, malformed JSON) yields a GameState with ``ammo=None`` and the
reason recorded in ``raw_response`` for the eval / reflection layers.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from .base import GameState, GameStatePerceptor

DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "vizdoom_ammo_v1.txt"
)


def encode_image_data_uri(screen_rgb: np.ndarray) -> str:
    """Encode an RGB screenshot as a base64 PNG data URI.

    Trajectory frames are RGB (ViZDoom layout, already transposed). cv2
    interprets arrays as BGR, so we convert first to keep the HUD colors
    correct in the PNG the VLM receives.
    """

    bgr = cv2.cvtColor(screen_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("cv2.imencode failed to encode screenshot")
    b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def parse_vlm_json(text: str) -> Tuple[Optional[dict], Optional[str]]:
    """Tolerantly extract the first JSON object from a VLM reply.

    Returns ``(parsed_dict, None)`` on success or ``(None, reason)`` on
    failure. Strips markdown fences and ignores any prose around the JSON.
    """

    if not text or not text.strip():
        return None, "empty response"

    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None, "no JSON object found"

    try:
        return json.loads(match.group(0)), None
    except json.JSONDecodeError as exc:
        return None, f"json decode error: {exc}"


class VLMPerceptor(GameStatePerceptor):
    def __init__(self, backend: Any, prompt_path: Optional[str] = None) -> None:
        self.backend = backend
        self.prompt = Path(prompt_path or DEFAULT_PROMPT_PATH).read_text(encoding="utf-8")

    def perceive(self, screenshot: np.ndarray, **kwargs: Any) -> GameState:
        try:
            data_uri = encode_image_data_uri(screenshot)
        except Exception as exc:  # noqa: BLE001 - no-raise contract
            return GameState(raw_response={"error": f"encode: {exc}"})

        try:
            response = self.backend.infer(self.prompt, data_uri)
        except Exception as exc:  # noqa: BLE001 - no-raise contract
            return GameState(raw_response={"error": f"backend: {exc}"})

        meta = {
            "raw_text": response.text,
            "latency_ms": response.latency_ms,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
        }

        parsed, reason = parse_vlm_json(response.text)
        if reason is not None:
            meta["error"] = reason
            return GameState(raw_response=meta)

        ammo = parsed.get("ammo")
        ammo_value = int(ammo) if isinstance(ammo, (int, float)) else None
        meta["vlm_level"] = parsed.get("ammo_level")
        return GameState(ammo=ammo_value, raw_response=meta)
