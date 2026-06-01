"""Unit tests for ``perception.ground_truth``.

Covers the ``ammo_level`` bucket boundaries (shared by GT and VLM scoring)
and ``GroundTruthPerceptor`` reading from recorded ``game_variables`` dicts.
No ViZDoom launch required.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from perception import GameState, GroundTruthPerceptor, ammo_level


class TestAmmoLevel:
    def test_high_boundary_inclusive(self):
        assert ammo_level(26) == "high"
        assert ammo_level(18) == "high"

    def test_medium_boundaries(self):
        assert ammo_level(17) == "medium"
        assert ammo_level(9) == "medium"

    def test_low_below_nine(self):
        assert ammo_level(8) == "low"
        assert ammo_level(0) == "low"

    def test_none_is_unknown(self):
        assert ammo_level(None) == "unknown"


class TestGroundTruthPerceptor:
    def test_reads_ammo_and_health_from_game_variables(self):
        gt = GroundTruthPerceptor()
        state = gt.perceive(game_variables={"ammo": 7, "health": 24})

        assert isinstance(state, GameState)
        assert state.ammo == 7
        assert state.health == 24
        assert state.raw_response["game_variables"] == {"ammo": 7, "health": 24}

    def test_missing_field_stays_none_without_raising(self):
        gt = GroundTruthPerceptor()
        state = gt.perceive(game_variables={"ammo": 12})

        assert state.ammo == 12
        assert state.health is None

    def test_empty_input_returns_all_none(self):
        gt = GroundTruthPerceptor()
        state = gt.perceive()

        assert state.ammo is None
        assert state.health is None

    def test_screenshot_is_ignored(self):
        gt = GroundTruthPerceptor()
        screen = np.zeros((4, 4, 3), dtype=np.uint8)
        state = gt.perceive(screen, game_variables={"ammo": 20})

        assert state.ammo == 20

    def test_fallback_to_live_state_with_var_names(self):
        gt = GroundTruthPerceptor()
        live = SimpleNamespace(game_variables=[15, 88], var_names=["ammo", "health"])
        state = gt.perceive(vizdoom_state=live)

        assert state.ammo == 15
        assert state.health == 88

    def test_live_state_without_var_names_is_ignored(self):
        gt = GroundTruthPerceptor()
        live = SimpleNamespace(game_variables=[15, 88])
        state = gt.perceive(vizdoom_state=live)

        assert state.ammo is None
        assert state.health is None
