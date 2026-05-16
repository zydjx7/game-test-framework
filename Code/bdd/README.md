# AssaultCube BDD Test Framework

This directory contains the active AssaultCube BDD flow:

```text
Code/bdd/run_tests.py -> behave features -> features/steps/weapon_steps.py -> Code/GameStateChecker Flask/LogicLayer
```

The old RiverGame pytest path is kept as legacy coverage and is not part of the default AssaultCube validation signal.

## Dependencies

```bash
pip install behave python-dotenv openai opencv-python
```

The project uses the OpenAI-compatible Python SDK to call DeepSeek, because DeepSeek exposes an OpenAI-compatible API format.

## DeepSeek Configuration

Use the workspace-root `.env` file as the single LLM configuration entrypoint:

```env
DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

`OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` are still accepted as legacy aliases so older local `.env` files do not break immediately, but new configuration should use `DEEPSEEK_*`.

Do not commit or paste API keys into chat. DeepSeek keys are normally shown only when created; if you cannot see an old key, create a new one in the DeepSeek platform and place it in the root `.env`.

To update the key without exposing it in command history:

```bash
python Code/bdd/update_api_key.py --prompt-key --test
```

To test the current configuration:

```bash
python Code/bdd/test_api_connection.py
```

## Proxy Note

For real DeepSeek calls, make sure `HTTP_PROXY`, `HTTPS_PROXY`, or `ALL_PROXY` do not point to `127.0.0.1:9` or `localhost:9`. That value is a blocking placeholder, not a running proxy server.

The shared LLM client bypasses only that invalid placeholder for DeepSeek requests. Valid proxy settings are left untouched.

## Running Tests

Run predefined AssaultCube tests:

```bash
python Code/bdd/run_tests.py --mode predefined --target assaultcube
```

Run a specific feature:

```bash
python Code/bdd/run_tests.py --mode predefined --feature generated_test.feature --target assaultcube
```

Generate and run one test case with DeepSeek:

```bash
python Code/bdd/run_tests.py --mode generated --req "test weapon switching" --target assaultcube
```

Generate and run multiple test cases with DeepSeek:

```bash
python Code/bdd/run_tests.py --mode batch --req "test weapon and ammo behavior" --count 3 --target assaultcube
```
