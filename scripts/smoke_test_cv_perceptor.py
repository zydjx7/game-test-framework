"""End-to-end smoke test for ``perception.CVPerceptor``.

Loads real PNGs from ``Code/GameStateChecker/test_images/`` and runs them
through ``CVPerceptor`` backed by the real (un-mocked) ``LogicLayer``.
This is the complement to the mocked unit tests in
``tests/test_cv_perceptor.py`` -- those prove the adapter logic is right,
this proves the wrapper actually composes with the legacy CV stack
(OpenCV + loguru + yaml + Tesseract + the template recogniser).

Usage::

    python scripts/smoke_test_cv_perceptor.py

Exit code is ``0`` if every ``perceive()`` call returned a ``GameState``
without raising, ``1`` otherwise. The script does **not** assert on
recognition accuracy: ``config.yaml`` is gitignored so the legacy ROI
coordinates fall back to ``p1_legacy`` defaults, which may misread some
AssaultCube frames. The point here is "does the wrapper run end-to-end",
not "does the legacy CV recognise correctly".
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from perception import CVPerceptor, GameState  # noqa: E402

TEST_IMAGES = REPO_ROOT / "Code" / "GameStateChecker" / "test_images"
CROSSHAIR_DIR = TEST_IMAGES / "Crosshair"
AMMO_DIR = TEST_IMAGES / "Ammo"


def _load(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"cv2.imread returned None for {path}")
    return img


def _fmt_state(state: GameState) -> str:
    """Compact one-line view; raw_response keys only (values can be huge)."""
    raw_keys = sorted(state.raw_response.keys())
    return (
        f"GameState(ammo={state.ammo!r}, "
        f"health={state.health!r}, "
        f"weapon={state.weapon!r}, "
        f"crosshair_red={state.crosshair_red!r}, "
        f"enemies_visible={state.enemies_visible!r}, "
        f"raw_response_keys={raw_keys})"
    )


def _expected_from_ammo_filename(name: str):
    """``ammo_clip17_total40.png`` -> 17."""
    m = re.search(r"clip(\d+)_total", name)
    return int(m.group(1)) if m else None


def _expected_from_crosshair_filename(name: str) -> bool:
    """Legacy convention from LogicLayer.__main__: dead/reload/teammate
    frames are *not* expected to show the red ready-state crosshair."""
    lower = name.lower()
    return not ("dead" in lower or "reload" in lower or "teammate" in lower)


def run_crosshair_case(perceptor: CVPerceptor, image_path: Path) -> bool:
    print(f"\n[crosshair] {image_path.name}")
    expected = _expected_from_crosshair_filename(image_path.name)
    try:
        screenshot = _load(image_path)
        state = perceptor.perceive(
            screenshot,
            check_crosshair=True,
            screenshot_path=str(image_path),
        )
    except Exception as exc:
        print(f"  RAISED: {exc!r}")
        return False
    match = "OK" if state.crosshair_red == expected else "MISMATCH"
    print(f"  expected={expected}, got={state.crosshair_red} [{match}]")
    print(f"  {_fmt_state(state)}")
    return True


def run_ammo_case(perceptor: CVPerceptor, image_path: Path) -> bool:
    expected = _expected_from_ammo_filename(image_path.name)
    if expected is None:
        return True
    print(f"\n[ammo] {image_path.name} (expected={expected})")
    try:
        screenshot = _load(image_path)
        state = perceptor.perceive(
            screenshot,
            expected_ammo=expected,
            screenshot_path=str(image_path),
        )
    except Exception as exc:
        print(f"  RAISED: {exc!r}")
        return False
    match = "OK" if state.ammo == expected else "MISMATCH"
    print(f"  recognised ammo={state.ammo} [{match}]")
    print(f"  {_fmt_state(state)}")
    return True


def main() -> int:
    if not TEST_IMAGES.exists():
        print(f"FATAL: test_images dir not found at {TEST_IMAGES}", file=sys.stderr)
        return 1

    print(f"REPO_ROOT = {REPO_ROOT}")
    print(f"TEST_IMAGES = {TEST_IMAGES}")
    print("Constructing CVPerceptor with real LogicLayer (target='assaultcube')...")
    try:
        perceptor = CVPerceptor(target_name="assaultcube")
    except Exception as exc:
        print(f"FATAL: CVPerceptor construction raised: {exc!r}", file=sys.stderr)
        return 1
    print(f"  -> {type(perceptor).__name__} wired; underlying logic={type(perceptor._logic).__name__}")

    # Pick a small representative sample so the script is fast and the
    # output stays readable. Order: one expected-true crosshair, one
    # expected-false crosshair, two ammo frames (single + double digit).
    crosshair_samples = [
        CROSSHAIR_DIR / "cross_normal.png",
        CROSSHAIR_DIR / "cross_reload.png",
    ]
    ammo_samples = [
        AMMO_DIR / "ammo_clip5_total40.png",
        AMMO_DIR / "ammo_clip17_total40.png",
    ]

    all_ran = True
    for p in crosshair_samples:
        if not p.exists():
            print(f"\n[crosshair] SKIP {p.name} (not found)")
            continue
        all_ran &= run_crosshair_case(perceptor, p)

    for p in ammo_samples:
        if not p.exists():
            print(f"\n[ammo] SKIP {p.name} (not found)")
            continue
        all_ran &= run_ammo_case(perceptor, p)

    # Final combined call to prove kwargs compose under real conditions.
    combined = AMMO_DIR / "ammo_clip5_total40.png"
    if combined.exists():
        print(f"\n[combined] {combined.name} (expected_ammo=5, check_crosshair=True)")
        try:
            state = perceptor.perceive(
                _load(combined),
                expected_ammo=5,
                check_crosshair=True,
                screenshot_path=str(combined),
            )
            print(f"  {_fmt_state(state)}")
        except Exception as exc:
            print(f"  RAISED: {exc!r}")
            all_ran = False

    print("\n=== SMOKE TEST {} ===".format("PASSED" if all_ran else "FAILED"))
    return 0 if all_ran else 1


if __name__ == "__main__":
    sys.exit(main())
