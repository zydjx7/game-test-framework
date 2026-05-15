"""CVPerceptor -- thin wrapper around the legacy AssaultCube CV pipeline.

The legacy ``Code/GameStateChecker/LogicLayer.py`` is **assertion-based**:
its methods take an expected value and return ``bool``. To fit the
**perception-based** :class:`~perception.base.GameStatePerceptor`
interface, this wrapper:

1. Calls the legacy assertion method with the provided hint.
2. Pulls the recognised value out of the ``context`` dict that
   ``LogicLayer`` writes into as a side effect (``template_result`` /
   ``ocr_result`` for ammo). This is how the legacy code already
   exposes the recognised number -- see ``LogicLayer.testAmmoTextInSync``.

Limitations inherited from AssaultCube CV (documented, not bugs):

- ``ammo`` requires an ``expected_ammo`` hint, because the legacy ROI
  picker switches between 1-digit and 2-digit boxes based on the
  expected value. Without the hint, ``ammo`` stays ``None``.
- ``health``, ``weapon``, ``enemies_visible`` are not implemented in
  AssaultCube CV; they always come back ``None`` here. The
  GroundTruth and VLM perceptors (Phase 1+) will fill them.

The wrapper accepts a pre-built ``logic_layer`` argument so unit tests
can inject a fake without importing OpenCV / loguru / yaml / Flask.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import GameState, GameStatePerceptor

# Code/GameStateChecker uses absolute imports internally (e.g.
# ``from VisionUtils import VisionUtils``), so we must add that directory
# to ``sys.path`` before importing ``LogicLayer``. We do this lazily
# inside ``__init__`` to keep ``import perception`` cheap and side-effect
# free for callers that only need ``GameState`` / the ABC.
_GSC_DIR = Path(__file__).resolve().parent.parent / "Code" / "GameStateChecker"


class CVPerceptor(GameStatePerceptor):
    """Adapter from the legacy CV pipeline to ``GameStatePerceptor``."""

    def __init__(
        self,
        target_name: str = "assaultcube",
        config: Optional[dict] = None,
        logic_layer: Any = None,
    ) -> None:
        """Construct a CV perceptor.

        Parameters
        ----------
        target_name:
            Forwarded to ``LogicLayer`` (e.g. ``"assaultcube"`` or
            ``"p1_legacy"``). Ignored when ``logic_layer`` is given.
        config:
            Optional config dict forwarded to ``LogicLayer``. When
            ``None`` the legacy class loads its own ``config.yaml``.
            Ignored when ``logic_layer`` is given.
        logic_layer:
            Pre-built ``LogicLayer`` (or test double). When provided the
            wrapper skips the real import entirely -- this keeps the
            unit tests free of OpenCV / loguru / yaml dependencies.
        """
        if logic_layer is not None:
            self._logic = logic_layer
            return

        gsc_dir = str(_GSC_DIR)
        if gsc_dir not in sys.path:
            sys.path.insert(0, gsc_dir)
        from LogicLayer import LogicLayer  # type: ignore[import-not-found]

        self._logic = LogicLayer(target_name=target_name, config=config)

    def perceive(self, screenshot: np.ndarray, **kwargs: Any) -> GameState:
        """Run the legacy CV pipeline against ``screenshot``.

        Recognised kwargs (all optional):

        - ``expected_ammo`` (int): triggers ``testAmmoTextInSync``; the
          recognised number (template > OCR) is written to ``ammo``.
        - ``check_crosshair`` (bool, default ``False``): triggers
          ``testWeaponCrossPresence`` and stores its bool result in
          ``crosshair_red``.
        - ``debug`` (bool, default ``False``) and ``debug_dir`` (str):
          forwarded to the legacy debug pipeline.
        - ``screenshot_path`` (str): passed into the legacy ``context``
          so its debug filenames stay meaningful.

        Failures inside ``LogicLayer`` are caught and recorded under
        ``raw_response``; the corresponding field stays ``None``. This
        preserves the no-raise contract of :meth:`GameStatePerceptor.perceive`.
        """
        raw: dict = {}
        ammo_val: Optional[int] = None
        crosshair_val: Optional[bool] = None

        expected_ammo = kwargs.get("expected_ammo")
        check_crosshair = kwargs.get("check_crosshair", False)
        debug = bool(kwargs.get("debug", False))
        debug_dir = kwargs.get("debug_dir")
        screenshot_path = kwargs.get("screenshot_path", "")

        if check_crosshair:
            context: dict = {"screenshotFile": screenshot_path}
            try:
                crosshair_val = bool(
                    self._logic.testWeaponCrossPresence(
                        [screenshot],
                        context,
                        {"boolResult": "True"},
                        debugEnabled=debug,
                        debug_dir=debug_dir,
                    )
                )
                raw["crosshair_context"] = context
            except Exception as exc:
                raw["crosshair_error"] = repr(exc)
                crosshair_val = None

        if expected_ammo is not None:
            context = {"screenshotFile": screenshot_path}
            try:
                self._logic.testAmmoTextInSync(
                    [screenshot],
                    context,
                    {"intResult": int(expected_ammo)},
                    debugEnabled=debug,
                    debug_dir=debug_dir,
                )
                ammo_val = _extract_ammo_from_context(context)
                raw["ammo_context"] = context
            except Exception as exc:
                raw["ammo_error"] = repr(exc)
                ammo_val = None

        return GameState(
            ammo=ammo_val,
            crosshair_red=crosshair_val,
            raw_response=raw,
        )


def _extract_ammo_from_context(context: dict) -> Optional[int]:
    """Pull the recognised ammo digit out of the legacy ``context`` dict.

    Prefers the template-matching result (``LogicLayer`` calls it more
    reliable for HUD digits) and falls back to OCR, stripping non-digits
    so noise like ``"  18 \\n"`` parses cleanly.
    """
    template_result = context.get("template_result")
    if template_result is not None:
        cleaned = "".join(ch for ch in str(template_result) if ch.isdigit())
        if cleaned:
            return int(cleaned)

    ocr_result = context.get("ocr_result")
    if ocr_result is not None:
        cleaned = "".join(ch for ch in str(ocr_result) if ch.isdigit())
        if cleaned:
            return int(cleaned)

    return None
