"""OpenAI-compatible POST helper + error translation.

Sits alongside the ``openai`` SDK calls in ``core.benchmark.openai``. We
keep these here (instead of in ``utils.http``) because the surface area
is intentionally narrow: a single rich exception type, the helper that
raises it, and the human-readable hint generator that translates 401/403
into "did the entrance auth flip succeed?" guidance.

The hint function recognises both raw ``urllib.error.HTTPError`` (raised
by the SDK's underlying httpx layer in some failure modes) AND our own
``OpenAIHTTPError``, so callers can pass through whatever they caught.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class OpenAIHTTPError(RuntimeError):
    """Wraps a non-JSON / non-2xx response with status + content-type +
    body snippet so JSONDecodeError doesn't swallow what we actually got
    (HTML login page, empty body, server-side error JSON, ...).
    """

    def __init__(self, message: str, *, status: int | None = None,
                 url: str = "", content_type: str = "", body: str = ""):
        super().__init__(message)
        self.status = status
        self.url = url
        self.content_type = content_type
        self.body = body


def _decode_body(resp_or_err) -> tuple[str, str]:
    """Returns (body_text, content_type). Best-effort; never raises."""
    content_type = ""
    try:
        content_type = (resp_or_err.headers.get("Content-Type", "") or "")
    except Exception:
        pass
    try:
        raw = resp_or_err.read()
    except Exception:
        return "", content_type
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8", errors="replace"), content_type
        except Exception:
            return raw.decode("latin-1", errors="replace"), content_type
    return str(raw), content_type


def post_openai(url: str, payload: dict, headers: dict,
                *, timeout: int) -> tuple[float, dict]:
    """POST `payload` as JSON. Returns (wall_seconds, parsed_json_body).

    Raises OpenAIHTTPError on 4xx/5xx, empty body, or 2xx with non-JSON
    body (HTML login page, SSE stream, ...).
    """
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers,
                                 method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw, content_type = _decode_body(resp)
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw, content_type = _decode_body(exc)
        snippet = raw.strip()[:300].replace("\n", " ")
        raise OpenAIHTTPError(
            f"HTTP {exc.code} from {url} "
            f"(content-type={content_type!r}): {snippet}",
            status=exc.code, url=url, content_type=content_type, body=raw,
        ) from None
    wall = time.perf_counter() - started

    if not raw.strip():
        raise OpenAIHTTPError(
            f"HTTP {status} from {url} returned an empty body",
            status=status, url=url, content_type=content_type, body=raw,
        )
    try:
        return wall, json.loads(raw)
    except json.JSONDecodeError as exc:
        snippet = raw.strip()[:300].replace("\n", " ")
        # Common gotchas: 1) HTML login page (Authelia / ingress not flipped
        # to public yet); 2) SSE stream because `stream:true` slipped into
        # the payload.
        looks_like_html = snippet.lstrip().lower().startswith(
            ("<!doctype", "<html"))
        looks_like_sse = snippet.startswith("data:")
        hint = ""
        if looks_like_html:
            hint = (" — looks like an HTML page; the entrance isn't "
                    "anonymous. Set `auto_open_internal_entrance:true` "
                    "or set `endpoint_url` to an in-cluster URL.")
        elif looks_like_sse:
            hint = (" — looks like an SSE stream; payload must include "
                    "`stream:false` for this script's parser.")
        raise OpenAIHTTPError(
            f"HTTP {status} from {url} returned non-JSON body "
            f"(content-type={content_type!r}): {snippet}{hint} "
            f"[json error: {exc.msg}]",
            status=status, url=url, content_type=content_type, body=raw,
        ) from None


def auth_hint(exc: Exception) -> str | None:
    """Friendly hint when an HTTP error looks like an auth problem.
    Recognizes both raw urllib HTTPError and our richer OpenAIHTTPError.
    """
    code: int | None = None
    if isinstance(exc, urllib.error.HTTPError):
        code = exc.code
    elif isinstance(exc, OpenAIHTTPError):
        code = exc.status
    if code in (401, 403):
        return (f"HTTP {code} — the entrance rejected the request. The "
                "script tries to flip authLevel to public automatically "
                "(see `auto_open_internal_entrance`); this likely means "
                "the flip never landed. Re-run, or run manually:\n"
                "  olares-cli settings apps auth-level set <app> <entrance> "
                "--level public\n"
                "  olares-cli settings apps policy set <app> <entrance> "
                "--default-policy public")
    return None


__all__ = ["OpenAIHTTPError", "auth_hint", "post_openai"]
