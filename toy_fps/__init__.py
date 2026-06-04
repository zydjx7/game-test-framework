"""ToyFPS — a tiny pure-Python second game adapter.

Purpose (Doc/v2-roadmap.md Stage 1c): prove the framework is PORTABLE, not just
architecturally portable. ToyFPS implements the same adapter contracts as the
ViZDoom line (a state with game_variables, a perceptor, an action library with
list_templates/run/check_expectation/DESCRIPTIONS), so the SAME agent layer
(`run_agent_loop`, `run_reflective_agent`, goal parser, reflection) runs on it
UNCHANGED. It also exercises the generalized schema beyond ammo (health, score)
and doubles as a fast, ViZDoom-free test fixture.
"""

from .adapter import ToyActions, ToyPerceptor, ToyPrimitives
from .game import ToyFPS

__all__ = ["ToyFPS", "ToyPrimitives", "ToyPerceptor", "ToyActions"]
