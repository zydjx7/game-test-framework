#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run AssaultCube BDD tests and optional DeepSeek-generated scenarios."""

import argparse
import atexit
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from behave.configuration import Configuration
from behave.runner import Runner
from loguru import logger
from test_generator.test_executor import TestExecutor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BDD_ROOT = Path(__file__).resolve().parent
CODE_ROOT = BDD_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.client_helpers import load_deepseek_config


LOGS_DIR = BDD_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger.add(str(LOGS_DIR / "run_tests.log"), level="DEBUG", rotation="1 MB")

API_KEY = load_deepseek_config().api_key


def _is_port_open(host: str = "localhost", port: int = 5000) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex((host, port)) == 0


def start_flask_server(target: str | None = None):
    """Start the local GameStateChecker Flask service if it is not running."""
    if _is_port_open():
        logger.info("GameStateChecker Flask server already appears to be running on port 5000")
        return None

    flask_logs_dir = LOGS_DIR / "flask"
    flask_logs_dir.mkdir(parents=True, exist_ok=True)
    server_path = CODE_ROOT / "GameStateChecker" / "main_flask_server.py"

    env = os.environ.copy()
    if target:
        env["GAMECHECK_TARGET"] = target
        logger.info("Using GameStateChecker target: {}", target)

    stdout = open(flask_logs_dir / "flask_stdout.log", "w", encoding="utf-8")
    stderr = open(flask_logs_dir / "flask_stderr.log", "w", encoding="utf-8")
    server = subprocess.Popen(
        [sys.executable, str(server_path)],
        stdout=stdout,
        stderr=stderr,
        env=env,
    )

    def cleanup() -> None:
        stdout.close()
        stderr.close()
        if server.poll() is None:
            logger.info("Stopping GameStateChecker Flask server")
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()

    atexit.register(cleanup)
    logger.info("Started GameStateChecker Flask server with PID {}", server.pid)

    time.sleep(5)
    if not _is_port_open():
        logger.error("GameStateChecker Flask server did not start on port 5000")
        if server.poll() is not None:
            logger.error("Server exited with code {}", server.returncode)
        return None

    return server


def run_predefined_tests(target: str | None = None, feature_file: str | None = None) -> bool:
    start_flask_server(target)

    if target:
        os.environ["GAMECHECK_TARGET"] = target

    if feature_file:
        feature_path = BDD_ROOT / "features" / feature_file
        if not feature_path.exists():
            logger.error("Feature file not found: {}", feature_path)
            return False
        paths = [str(feature_path)]
    else:
        paths = [str(BDD_ROOT / "features")]

    config = Configuration(command_args=[])
    config.paths = paths
    config.format = ["pretty"]
    result = Runner(config).run()
    return result == 0


def run_generated_tests(
    api_key: str | None = None,
    requirement: str | None = None,
    target: str | None = None,
) -> bool:
    if not requirement:
        requirement = "Generate an AssaultCube weapon interaction test."

    start_flask_server(target)
    executor = TestExecutor(api_key, target=target)
    start_time = time.time()
    result = executor.execute_from_requirement(requirement)
    report = executor.generate_report(result, time.time() - start_time)
    logger.info("Generated test report: {}", report)
    print(report)
    return report["status"] == "success"


def run_batch_tests(
    api_key: str | None = None,
    requirement: str | None = None,
    count: int = 5,
    target: str | None = None,
) -> bool:
    if not requirement:
        requirement = "Generate several AssaultCube weapon and ammo behavior tests."

    start_flask_server(target)
    executor = TestExecutor(api_key, target=target)
    start_time = time.time()
    result = executor.execute_batch(requirement, count)
    report = executor.generate_report(result, time.time() - start_time)
    logger.info("Batch generated test report: {}", report)
    print(report)
    return report["status"] == "success"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AssaultCube BDD tests")
    parser.add_argument("--api-key", dest="api_key", help="DeepSeek API key")
    parser.add_argument("--req", dest="requirement", help="Test generation requirement")
    parser.add_argument("--count", type=int, default=5, help="Number of generated test cases")
    parser.add_argument("--target", dest="target", help="GameStateChecker target, e.g. assaultcube")
    parser.add_argument(
        "--mode",
        choices=["predefined", "generated", "batch"],
        default="predefined",
        help="Test mode",
    )
    parser.add_argument(
        "--feature",
        dest="feature_file",
        help="Feature file name under Code/bdd/features for predefined mode",
    )
    parser.add_argument(
        "--use-llm-analysis",
        action="store_true",
        help="Use DeepSeek for auxiliary in-test analysis steps",
    )
    args = parser.parse_args()

    os.environ["USE_LLM_ANALYSIS"] = "true" if args.use_llm_analysis else "false"
    api_key = args.api_key or API_KEY

    if args.mode == "predefined":
        ok = run_predefined_tests(args.target, args.feature_file)
    elif args.mode == "generated":
        ok = run_generated_tests(api_key, args.requirement, args.target)
    else:
        ok = run_batch_tests(api_key, args.requirement, args.count, args.target)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
