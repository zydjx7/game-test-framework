"""Unit tests for ``perception.CVPerceptor``.

The legacy ``LogicLayer`` is mocked via the ``logic_layer=`` injection
slot, so these tests do **not** touch OpenCV, loguru, yaml, Flask, or
the legacy template/OCR pipelines. They only verify the adapter
behaviour: which kwargs trigger which legacy call, how recognised
values are extracted from the legacy ``context`` dict, and how
exceptions are absorbed without breaking the no-raise contract of
``GameStatePerceptor.perceive``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from perception import CVPerceptor, GameState, GameStatePerceptor


class FakeLogicLayer:
    """Minimal in-memory stand-in for ``Code.GameStateChecker.LogicLayer``.

    Records every call and lets each test tweak return values + the
    side-effects that the real class writes into the ``context`` dict
    (``template_result`` and ``ocr_result``).
    """

    def __init__(self) -> None:
        self.crosshair_calls: List[Tuple[Any, Dict, Dict, Dict]] = []
        self.ammo_calls: List[Tuple[Any, Dict, Dict, Dict]] = []
        self.crosshair_return: bool = True
        self.ammo_match_return: bool = True
        self.template_result: Any = None
        self.ocr_result: Any = None

    def testWeaponCrossPresence(
        self, screenshots, context, expected, **kwargs
    ):
        self.crosshair_calls.append((screenshots, context, expected, kwargs))
        return self.crosshair_return

    def testAmmoTextInSync(self, screenshots, context, expected, **kwargs):
        self.ammo_calls.append((screenshots, context, expected, kwargs))
        if self.template_result is not None:
            context["template_result"] = self.template_result
        if self.ocr_result is not None:
            context["ocr_result"] = self.ocr_result
        return self.ammo_match_return


@pytest.fixture
def fake_screenshot() -> np.ndarray:
    """Synthetic BGR frame -- contents do not matter (LogicLayer is mocked)."""
    return np.zeros((64, 64, 3), dtype=np.uint8)


def test_gamestate_defaults_are_all_none():
    state = GameState()
    assert state.ammo is None
    assert state.health is None
    assert state.weapon is None
    assert state.crosshair_red is None
    assert state.enemies_visible is None
    assert state.raw_response == {}


def test_cv_perceptor_is_a_game_state_perceptor():
    perc = CVPerceptor(logic_layer=FakeLogicLayer())
    assert isinstance(perc, GameStatePerceptor)


def test_perceive_with_no_hints_returns_empty_state_and_calls_nothing(
    fake_screenshot,
):
    fake = FakeLogicLayer()
    perc = CVPerceptor(logic_layer=fake)

    state = perc.perceive(fake_screenshot)

    assert isinstance(state, GameState)
    assert state.ammo is None
    assert state.crosshair_red is None
    assert state.health is None
    assert state.weapon is None
    assert state.enemies_visible is None
    assert state.raw_response == {}
    assert fake.crosshair_calls == []
    assert fake.ammo_calls == []


def test_perceive_with_expected_ammo_extracts_template_result(fake_screenshot):
    fake = FakeLogicLayer()
    fake.template_result = "42"
    perc = CVPerceptor(logic_layer=fake)

    state = perc.perceive(fake_screenshot, expected_ammo=42)

    assert state.ammo == 42
    assert len(fake.ammo_calls) == 1
    _screens, _ctx, expected, _kwargs = fake.ammo_calls[0]
    assert expected == {"intResult": 42}


def test_perceive_prefers_template_over_ocr(fake_screenshot):
    fake = FakeLogicLayer()
    fake.template_result = "9"
    fake.ocr_result = "99"
    perc = CVPerceptor(logic_layer=fake)

    state = perc.perceive(fake_screenshot, expected_ammo=9)

    assert state.ammo == 9


def test_perceive_falls_back_to_ocr_when_template_missing(fake_screenshot):
    fake = FakeLogicLayer()
    fake.template_result = None
    fake.ocr_result = "  18 \n"
    perc = CVPerceptor(logic_layer=fake)

    state = perc.perceive(fake_screenshot, expected_ammo=18)

    assert state.ammo == 18


def test_perceive_ammo_none_when_both_recognisers_fail(fake_screenshot):
    fake = FakeLogicLayer()
    fake.template_result = None
    fake.ocr_result = ""
    fake.ammo_match_return = False
    perc = CVPerceptor(logic_layer=fake)

    state = perc.perceive(fake_screenshot, expected_ammo=5)

    assert state.ammo is None
    assert "ammo_context" in state.raw_response


def test_perceive_with_check_crosshair_true(fake_screenshot):
    fake = FakeLogicLayer()
    fake.crosshair_return = True
    perc = CVPerceptor(logic_layer=fake)

    state = perc.perceive(fake_screenshot, check_crosshair=True)

    assert state.crosshair_red is True
    assert len(fake.crosshair_calls) == 1
    assert fake.ammo_calls == []


def test_perceive_with_check_crosshair_false(fake_screenshot):
    fake = FakeLogicLayer()
    fake.crosshair_return = False
    perc = CVPerceptor(logic_layer=fake)

    state = perc.perceive(fake_screenshot, check_crosshair=True)

    assert state.crosshair_red is False


def test_perceive_combined_kwargs(fake_screenshot):
    fake = FakeLogicLayer()
    fake.crosshair_return = True
    fake.template_result = "7"
    perc = CVPerceptor(logic_layer=fake)

    state = perc.perceive(
        fake_screenshot, expected_ammo=7, check_crosshair=True
    )

    assert state.crosshair_red is True
    assert state.ammo == 7
    assert len(fake.crosshair_calls) == 1
    assert len(fake.ammo_calls) == 1


def test_perceive_swallows_crosshair_exception(fake_screenshot):
    fake = FakeLogicLayer()

    def boom(*_a, **_kw):
        raise RuntimeError("simulated crosshair failure")

    fake.testWeaponCrossPresence = boom  # type: ignore[assignment]
    perc = CVPerceptor(logic_layer=fake)

    state = perc.perceive(fake_screenshot, check_crosshair=True)

    assert state.crosshair_red is None
    assert "crosshair_error" in state.raw_response
    assert "simulated" in state.raw_response["crosshair_error"]


def test_perceive_swallows_ammo_exception(fake_screenshot):
    fake = FakeLogicLayer()

    def boom(*_a, **_kw):
        raise ValueError("simulated ammo failure")

    fake.testAmmoTextInSync = boom  # type: ignore[assignment]
    perc = CVPerceptor(logic_layer=fake)

    state = perc.perceive(fake_screenshot, expected_ammo=10)

    assert state.ammo is None
    assert "ammo_error" in state.raw_response


def test_perceive_does_not_mutate_input_screenshot(fake_screenshot):
    fake = FakeLogicLayer()
    perc = CVPerceptor(logic_layer=fake)
    snapshot = fake_screenshot.copy()

    perc.perceive(fake_screenshot, expected_ammo=1, check_crosshair=True)

    assert np.array_equal(fake_screenshot, snapshot)


def test_perceive_forwards_screenshot_path_into_context(fake_screenshot):
    fake = FakeLogicLayer()
    perc = CVPerceptor(logic_layer=fake)

    perc.perceive(
        fake_screenshot,
        expected_ammo=3,
        check_crosshair=True,
        screenshot_path="some/file.png",
    )

    assert fake.crosshair_calls[0][1]["screenshotFile"] == "some/file.png"
    assert fake.ammo_calls[0][1]["screenshotFile"] == "some/file.png"


def test_perceive_forwards_debug_flags(fake_screenshot):
    fake = FakeLogicLayer()
    perc = CVPerceptor(logic_layer=fake)

    perc.perceive(
        fake_screenshot,
        expected_ammo=1,
        check_crosshair=True,
        debug=True,
        debug_dir="/tmp/dbg",
    )

    assert fake.crosshair_calls[0][3] == {
        "debugEnabled": True,
        "debug_dir": "/tmp/dbg",
    }
    assert fake.ammo_calls[0][3] == {
        "debugEnabled": True,
        "debug_dir": "/tmp/dbg",
    }
