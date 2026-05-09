"""HTTP helpers used by readiness probes and the Ollama benchmark.

The OpenAI-compatible benchmark uses the `openai` SDK directly, so this
module only needs:
  - small urllib wrappers for /api/tags, /api/generate, /v1/models polling
  - a shared `auth_hint()` that turns a 401/403 from any backend into a
    Olares-specific "flip the entrance to public" hint
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Optional

log = logging.getLogger("llm_bench")


def http_post_json(url: str, payload: dict, *, timeout: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_get_status(url: str, *, timeout: int = 10):
    """GET `url`. Returns (status, body); (0, "") on transport-level failure.
    The readiness probe wants to retry on ANY non-2xx, including
    ConnectionRefused / Timeout / 502/503, so transport errors become status=0.
    """
    req = urllib.request.Request(
        url, method="GET", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            try:
                body = resp.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                body = ""
            return resp.status, body
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            body = ""
        return exc.code, body
    except Exception:  # noqa: BLE001
        return 0, ""


def ollama_tags(url: str, *, timeout: int = 10):
    """GET /api/tags. Returns (models_or_None, status, error_msg).

    `models_or_None=None` means "couldn't talk to the daemon at all".
    A reachable daemon with NO models pulled returns ([], 200, "").
    """
    status, body = http_get_status(f"{url.rstrip('/')}/api/tags",
                                   timeout=timeout)
    if not (200 <= status < 300):
        return (None, status, f"HTTP {status}")
    try:
        models = json.loads(body or "{}").get("models")
    except json.JSONDecodeError:
        return (None, status, "200 but body not JSON")
    if not isinstance(models, list):
        return (None, status, "200 but no models[] array")
    return (models, status, "")


def ollama_model_names(models: list) -> list:
    """Extract canonical names from /api/tags's models[]; coalesces
    name/model so we don't break if upstream drops one of them.
    """
    out: list = []
    for m in models or []:
        if not isinstance(m, dict):
            continue
        name = m.get("name") or m.get("model") or ""
        if name:
            out.append(name)
    return out


def auth_hint(exc: Exception) -> Optional[str]:
    """Return a friendly hint when an HTTP error looks like an auth issue.

    Recognizes:
      - urllib.error.HTTPError (raw urllib, used by Ollama)
      - openai.APIStatusError  (SDK, used by the OpenAI benchmark)
    """
    code: Optional[int] = None
    if isinstance(exc, urllib.error.HTTPError):
        code = exc.code
    else:
        # `openai` is a hard dep but only used by the OpenAI backend; keep
        # the import lazy so this module can be loaded without it.
        try:
            from openai import APIStatusError
        except ImportError:
            APIStatusError = None
        if APIStatusError is not None and isinstance(exc, APIStatusError):
            code = exc.status_code
    if code in (401, 403):
        return (f"HTTP {code} — the entrance rejected the request. The "
                "script tries to flip authLevel to public automatically "
                "(see `auto_open_internal_entrance`); this likely means "
                "the flip never landed or the policy default-policy is "
                "still non-public. Re-run, or run manually:\n"
                "  olares-cli settings apps auth-level set <app> <entrance> "
                "--level public\n"
                "  olares-cli settings apps policy set <app> <entrance> "
                "--default-policy public")
    return None
