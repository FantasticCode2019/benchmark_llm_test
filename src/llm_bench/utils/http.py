"""Generic HTTP helpers — backend-agnostic GET / POST / JSON.

Anything backend-specific (Ollama bundle endpoints, vLLM bundle SSE,
OpenAI-compatible error translation) lives in ``llm_bench.clients.*``;
this module is intentionally small so it has no `clients.*` imports and
can be reused by them.

Behaviour contract:

* ``http_get_status``  — returns (status, body); transport errors and
  non-2xx collapse to status=0 / status=<code> with body filled when the
  server actually sent one. Never raises.
* ``http_post_json``   — raises whatever urllib raises. Callers wrap.
* ``http_get_json``    — returns (parsed_dict_or_None, status) and DOES
  surface non-2xx JSON envelopes (some servers signal "model still
  loading" via a 503 with a JSON error body).
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from llm_bench.constants import HTTP_DEFAULT_TIMEOUT_SECONDS, LOG_NAMESPACE

log = logging.getLogger(LOG_NAMESPACE)


def http_post_json(url: str, payload: dict, *, timeout: int) -> dict:
    """POST a JSON payload, return parsed JSON body. Raises on any
    failure — callers translate to their domain-specific error.
    """
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_get_status(url: str, *,
                    timeout: int = HTTP_DEFAULT_TIMEOUT_SECONDS,
                    ) -> tuple[int, str]:
    """GET `url`. Returns (status, body); (0, "") on transport-level failure.

    Readiness probes want to retry on ANY non-2xx (ConnectionRefused,
    Timeout, 502/503, ...), so transport errors collapse to status=0.
    """
    req = urllib.request.Request(
        url, method="GET", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            try:
                body = resp.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            return resp.status, body
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return exc.code, body
    except Exception:
        return 0, ""


def http_get_json(url: str, *,
                  timeout: int = HTTP_DEFAULT_TIMEOUT_SECONDS,
                  ) -> tuple[dict | None, int]:
    """GET <url>. Returns (parsed_dict_or_None, http_status).

    Unlike the bundle helpers in ``llm_bench.clients.*`` this DOES
    surface non-2xx bodies when they parse as JSON, because "model still
    loading" is signalled by a 503 with a JSON envelope:
        {"error":{"message":"Loading model","type":"unavailable_error","code":503}}
    """
    status, body = http_get_status(url, timeout=timeout)
    try:
        data = json.loads(body or "{}")
    except json.JSONDecodeError:
        return (None, status)
    return ((data if isinstance(data, dict) else None), status)


__all__ = ["http_get_json", "http_get_status", "http_post_json"]
