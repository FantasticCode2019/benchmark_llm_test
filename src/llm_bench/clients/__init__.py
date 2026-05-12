"""Backend-specific HTTP clients (extracted from utils/http.py in Phase 2).

Each sub-module owns the wire vocabulary of one backend so utils/http.py
stays a thin, generic GET/POST/JSON layer:

    ollama_client   — /api/ps, /api/tags, /api/show, /api/progress
                      (descriptor merging, capability probing, readiness
                       progress snapshot)
    vllm_client     — /cfg, /progress?id=...  (multi-shape bundle config
                      resolver + SSE-aware progress poller)
    openai_errors   — OpenAIHTTPError + post_openai + auth_hint
                      (rich error wrapping for 4xx/5xx/non-JSON bodies on
                       the OpenAI-compatible POST path)
"""
