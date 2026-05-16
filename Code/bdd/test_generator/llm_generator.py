import os
import sys

from loguru import logger
from openai import OpenAI

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.llm.client_helpers import create_openai_client, load_deepseek_config


logger.add("logs/test_generator.log", level="DEBUG", rotation="1 MB")


class TestGenerator:
    def __init__(self, api_key=None):
        config = load_deepseek_config(api_key=api_key)
        if not config.api_key:
            logger.error("No DeepSeek API key found. Set DEEPSEEK_API_KEY in the root .env.")
            raise ValueError("No DeepSeek API key found")

        self.api_key = config.api_key
        self.model = config.model
        self.base_url = config.base_url
        self.client = create_openai_client(OpenAI, api_key=self.api_key, base_url=self.base_url)

        logger.info("Using DeepSeek model: {}", self.model)
        logger.info("Using DeepSeek base URL: {}", self.base_url)

    def create_test_prompt(self, test_requirement):
        use_llm_analysis = os.getenv("USE_LLM_ANALYSIS", "false").lower() == "true"
        ammo_related = any(
            word in test_requirement.lower()
            for word in ["ammo", "bullet", "fire", "shoot", "reload", "clip"]
        )

        allowed_steps = [
            "Given the game is started",
            "When player equips a weapon",
            "When player equips a primary weapon",
            "When player fires the weapon",
            "When player aims at an enemy",
            "When player aims at a teammate",
            "When player switches to grenade",
            "When player switches to knife",
            "When player switches to secondary weapon",
            "When player reloads the weapon",
            "When player dies",
            "Then the crosshair should be visible",
            "Then the crosshair should indicate aiming at an enemy",
            "Then the crosshair should indicate aiming at a teammate",
            "Then the crosshair should be for a grenade",
            "Then the crosshair should be for a knife",
            "Then the crosshair should be for a secondary weapon",
            "Then the crosshair should be for a primary weapon",
            "Then the crosshair should indicate reloading",
            "Then the player view should indicate death",
        ]

        if use_llm_analysis:
            allowed_steps.extend(
                [
                    "Then the crosshair should match the current weapon state",
                    "Then the crosshair should be analyzed by LLM",
                ]
            )

        if ammo_related:
            allowed_steps.extend(
                [
                    "Given the player has {number} ammo",
                    "Then the ammo count should decrease",
                    "Then the ammo count should match the expected value",
                    "Then the ammo displayed should be {number}",
                    "Then the ammo count should be verified by LLM",
                ]
            )

        step_list = "\n".join(f"- {step}" for step in allowed_steps)
        return f"""
Generate one Gherkin feature for this AssaultCube test requirement:
{test_requirement}

Only use these predefined steps:
{step_list}

Rules:
- Output only valid Gherkin, with Feature: and Scenario: keywords.
- Do not add Markdown fences or explanations.
- The framework validates static screenshots, not live game control.
- Prefer weapon/crosshair checks unless ammo is explicitly requested.
- AssaultCube primary weapon clip values are normally in the 0-20 range.
"""

    def generate_test_case(self, requirement):
        try:
            logger.info("Generating one test case with DeepSeek for: {}", requirement[:80])
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You generate concise Gherkin BDD tests for an AssaultCube "
                            "screenshot validation framework."
                        ),
                    },
                    {"role": "user", "content": self.create_test_prompt(requirement)},
                ],
            )
            test_case = response.choices[0].message.content
            logger.info("Generated test case with length {}", len(test_case))
            return test_case
        except Exception as exc:
            logger.error("Failed to generate test case with DeepSeek: {}", exc)
            return None

    def generate_multiple_test_cases(self, requirement, count=5):
        try:
            prompt = (
                self.create_test_prompt(requirement)
                + f"\nGenerate {count} distinct Scenario sections under one Feature."
            )
            logger.info("Generating {} test cases with DeepSeek for: {}", count, requirement[:80])
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You generate concise Gherkin BDD tests for an AssaultCube "
                            "screenshot validation framework."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            test_cases = response.choices[0].message.content
            logger.info("Generated batch test case text with length {}", len(test_cases))
            return test_cases
        except Exception as exc:
            logger.error("Failed to generate batch test cases with DeepSeek: {}", exc)
            return None
