"""HTTP helpers for the readiness probes and the benchmark POSTs.

Also hosts OpenAIHTTPError + auth_hint because both are about translating
a raw HTTP failure into something a human can read in the report.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional, Tuple

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


def http_get_status(url: str, *, timeout: int = 10) -> Tuple[int, str]:
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


def bundle_cfg(url: str, *,
               timeout: int = 10) -> Tuple[Optional[dict], int, str]:
    """GET <base>/cfg. Returns (parsed_dict_or_None, http_status, raw_body).

    The bundle-aware chart exposes /cfg with the running download-job
    descriptor (typically a top-level {jobId, probeUrl, probeIntervalMs,
    modelName, repo, ref, appUrl, ...}; legacy multi-task charts may
    instead expose {tasks:[{jobId,repo,file,...}], jobIds, probeUrl,
    probeIntervalMs}). The raw body is returned alongside so callers can
    log exactly what came back when parsing fails.
    """
    status, body = http_get_status(f"{url.rstrip('/')}/cfg", timeout=timeout)
    if not (200 <= status < 300):
        return (None, status, body)
    try:
        data = json.loads(body or "{}")
    except json.JSONDecodeError:
        return (None, status, body)
    return ((data if isinstance(data, dict) else None), status, body)


def ollama_progress(url: str, *,
                    timeout: int = 10) -> Tuple[Optional[dict], int, str]:
    """GET <base>/api/progress for **ollama** bundles (no query string).

    The ollama-side chart exposes /api/progress as a plain-JSON snapshot
    endpoint that returns the CURRENT job state on every call, e.g.
        {"app_url":"...","status":"downloading","model_name":"...",
         "progress":45.3,"speed_bps":125829120.0,"duration":12, ...}
    There is no /cfg step and no jobId parameter — ollama serves a
    single bundle at a time. Returns (parsed_dict_or_None, http_status,
    raw_body); raw body is forwarded so the readiness poller can log
    non-JSON / non-2xx responses verbatim.
    """
    status, body = http_get_status(
        f"{url.rstrip('/')}/api/progress", timeout=timeout)
    if not (200 <= status < 300):
        return (None, status, body)
    try:
        data = json.loads(body or "{}")
    except json.JSONDecodeError:
        return (None, status, body)
    return ((data if isinstance(data, dict) else None), status, body)


def vllm_progress(url: str, job_id: str, *,
                  timeout: int = 10) -> Tuple[Optional[dict], int, str]:
    """GET <base>/progress?id=<job_id> for **vllm** bundles.

    The vllm-side chart serves /progress as Server-Sent Events:
        data: {"id":"...","status":"running","downloaded":...}\n
        \n
        data: {"id":"...","status":"done","downloaded":...}\n

    We GET once, scan for `data: {json}` events, and return the LAST
    parsed event (= latest known state). Plain-JSON bodies are also
    accepted as a single snapshot. Returns
    (last_event_or_None, http_status, raw_body); non-2xx / nothing
    parseable returns (None, status, raw).
    """
    qid = urllib.parse.quote(job_id, safe="")
    status, body = http_get_status(
        f"{url.rstrip('/')}/progress?id={qid}", timeout=timeout)
    if not (200 <= status < 300):
        return (None, status, body)

    # Snapshot mode: whole body is plain JSON (no SSE framing).
    try:
        parsed = json.loads(body or "")
        if isinstance(parsed, dict) and parsed:
            return (parsed, status, body)
    except json.JSONDecodeError:
        pass

    # SSE mode: one event = one or more contiguous `data:` lines,
    # terminated by a blank line. We accumulate `data:` payloads and
    # parse them on the blank-line boundary (or at EOF), keeping the
    # LAST successfully-parsed event.
    last: Optional[dict] = None
    buf: list = []

    def _flush():
        nonlocal last
        if not buf:
            return
        blob = "".join(buf)
        buf.clear()
        try:
            evt = json.loads(blob)
        except json.JSONDecodeError:
            return
        if isinstance(evt, dict):
            last = evt

    for line in (body or "").splitlines():
        stripped = line.rstrip("\r")
        if not stripped.strip():
            _flush()
            continue
        if stripped.startswith("data:"):
            buf.append(stripped[5:].lstrip())
    _flush()

    return (last, status, body)


def ollama_tags(url: str, *,
                timeout: int = 10) -> Tuple[Optional[list], int, str]:
    """GET /api/tags. Returns (models_or_None, status, error_msg).

    `models_or_None=None` means the daemon was unreachable (transport
    failure, non-2xx, or non-JSON body). A reachable daemon with no
    models pulled yet returns ([], 200, "").
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
    `name`/`model` so we don't break if upstream renames one of them.
    """
    out: list = []
    for m in models or []:
        if not isinstance(m, dict):
            continue
        name = m.get("name") or m.get("model") or ""
        if name:
            out.append(name)
    return out


def http_get_json(url: str, *,
                  timeout: int = 10) -> Tuple[Optional[dict], int]:
    """GET <url>. Returns (parsed_dict_or_None, http_status).

    Unlike bundle_cfg / ollama_progress / vllm_progress this DOES
    surface non-2xx bodies
    when they parse as JSON, because "model still loading" is signalled
    by a 503 with a JSON envelope:
        {"error":{"message":"Loading model","type":"unavailable_error","code":503}}
    """
    status, body = http_get_status(url, timeout=timeout)
    try:
        data = json.loads(body or "{}")
    except json.JSONDecodeError:
        return (None, status)
    return ((data if isinstance(data, dict) else None), status)


class OpenAIHTTPError(RuntimeError):
    """Wraps a non-JSON / non-2xx response with status + content-type +
    body snippet so JSONDecodeError doesn't swallow what we actually got
    (HTML login page, empty body, server-side error JSON, ...).
    """

    def __init__(self, message: str, *, status: Optional[int] = None,
                 url: str = "", content_type: str = "", body: str = ""):
        super().__init__(message)
        self.status = status
        self.url = url
        self.content_type = content_type
        self.body = body


def _decode_body(resp_or_err) -> Tuple[str, str]:
    """Returns (body_text, content_type). Best-effort; never raises."""
    content_type = ""
    try:
        content_type = (resp_or_err.headers.get("Content-Type", "") or "")
    except Exception:  # noqa: BLE001
        pass
    try:
        raw = resp_or_err.read()
    except Exception:  # noqa: BLE001
        return "", content_type
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8", errors="replace"), content_type
        except Exception:  # noqa: BLE001
            return raw.decode("latin-1", errors="replace"), content_type
    return str(raw), content_type


def post_openai(url: str, payload: dict, headers: dict,
                *, timeout: int) -> Tuple[float, dict]:
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


def auth_hint(exc: Exception) -> Optional[str]:
    """Friendly hint when an HTTP error looks like an auth problem.
    Recognizes both raw urllib HTTPError and our richer OpenAIHTTPError.
    """
    code: Optional[int] = None
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
