"""Abstract perceptor interface + unified GameState dataclass.

Design intent (Phase 0.1):
- ``GameState`` is the *single* return type all perceptor backends must
  produce. Different backends fill different subsets of fields; unknown
  fields stay ``None``. Downstream code (oracle, reflection, agent loop)
  must treat ``None`` as "this backend did not extract it" rather than
  "this field has no value in the game".
- ``GameStatePerceptor.perceive`` accepts ``**kwargs`` so each backend
  can declare its own optional hints without polluting the shared
  signature (e.g. the CV backend needs an ``expected_ammo`` hint because
  its OCR ROI sizing depends on digit count; the ViZDoom GT backend
  needs the live ``vizdoom_state``; the VLM backend may take a custom
  prompt or backend selector).

Forward-compatibility for Phase 1+:
- ``perception/ground_truth.py`` will add ``GroundTruthPerceptor`` reading
  ``vizdoom_state.game_variables`` directly -- the kwarg shape is
  documented below so it is wired in without re-touching this file.
- ``perception/vlm_perceptor.py`` will add ``VLMPerceptor`` dispatching
  over DeepSeek-VL2 / Qwen-VL / GPT-4o.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class GameState:
    """Structured per-frame game state shared across perceptor backends.

    Every field is ``Optional`` because each backend fills a different
    subset:

    +---------------------------+--------------------------------------------+
    | Backend                   | Fields it can fill                         |
    +===========================+============================================+
    | ``CVPerceptor``           | ``ammo`` (with ``expected_ammo`` hint),    |
    | (AssaultCube baseline)    | ``crosshair_red``                          |
    +---------------------------+--------------------------------------------+
    | ``GroundTruthPerceptor``  | all numeric fields directly from           |
    | (ViZDoom, Phase 1+)       | ``state.game_variables`` -- treated as     |
    |                           | oracle for accuracy measurement            |
    +---------------------------+--------------------------------------------+
    | ``VLMPerceptor``          | all fields parsed from VLM JSON output     |
    | (Phase 1+)                |                                            |
    +---------------------------+--------------------------------------------+

    ``raw_response`` is reserved for debugging / evaluation: a backend may
    stash the underlying detector output (CV context dict, VLM JSON,
    raw ``game_variables`` array, ...) so experiments can diff against
    ground truth without re-running perception.
    """

    ammo: Optional[int] = None
    health: Optional[int] = None
    weapon: Optional[str] = None
    crosshair_red: Optional[bool] = None
    enemies_visible: Optional[bool] = None
    score: Optional[int] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


class GameStatePerceptor(ABC):
    """Abstract base class for all perceptor backends.

    Implementations promise:

    1. ``perceive`` does **not** raise on partial failure. If a field
       cannot be extracted (OCR failed, VLM returned malformed JSON, GT
       variable not configured), the corresponding ``GameState`` field
       stays ``None`` and the error is logged into ``raw_response`` for
       inspection. This lets the upstream agent loop / reflection module
       (Phase 3) decide how to react instead of catching exceptions
       everywhere.
    2. ``perceive`` is pure with respect to ``screenshot`` -- it does
       not mutate the input array.
    3. Backend-specific configuration (model name, target name, ViZDoom
       handle, ...) is passed at construction time, not per call.

    Recognised ``**kwargs`` per backend (Phase 1+ entries are forward
    declarations -- they go live when the corresponding file lands):

    - ``CVPerceptor``:
        * ``expected_ammo`` (int) -- hint for legacy OCR ROI sizing
        * ``check_crosshair`` (bool, default False) -- enable crosshair
          detection
        * ``debug`` (bool), ``debug_dir`` (str) -- forwarded to legacy
          debug pipeline
    - ``GroundTruthPerceptor`` (Phase 1):
        * ``vizdoom_state`` (vzd.GameState) -- live ViZDoom state object
    - ``VLMPerceptor`` (Phase 1):
        * ``prompt_override`` (str) -- override the default prompt
        * ``backend`` (str) -- per-call backend switch
    """

    @abstractmethod
    def perceive(self, screenshot: np.ndarray, **kwargs: Any) -> GameState:
        """Convert a single screenshot into a :class:`GameState`.

        Parameters
        ----------
        screenshot:
            ``(H, W, 3)`` BGR image in OpenCV layout.

            .. warning::
               ViZDoom's ``state.screen_buffer`` is ``(C, H, W)``
               channels-first. Callers must transpose with
               ``np.transpose(buf, (1, 2, 0))`` (and convert RGB->BGR if
               needed) before handing it to a perceptor. The contract
               here is OpenCV-style so the same screenshot can flow
               through any backend unchanged.
        **kwargs:
            Backend-specific hints (see class docstring).

        Returns
        -------
        GameState
            Fields not extracted by this backend are ``None``.
        """
