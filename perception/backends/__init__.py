"""VLM backend adapters for the perception layer.

Each backend wraps one provider behind a uniform ``infer(prompt, image)``
call returning a :class:`BackendResponse`. Phase 1 ships only
``Qwen3VLFlashBackend``; Phase 4 adds Gemini / Qwen3-VL-Plus / local
Qwen2.5-VL under the same protocol.
"""
