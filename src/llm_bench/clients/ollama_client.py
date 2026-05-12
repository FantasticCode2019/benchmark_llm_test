"""Ollama HTTP client.

Wraps the four Ollama endpoints we care about:

* ``GET  /api/progress``   bundle download snapshot (readiness probe)
* ``GET  /api/tags``       installed model catalogue
* ``GET  /api/ps``         currently-loaded model(s) — fetched implicitly
                           by the descriptor helpers
* ``POST /api/show``       per-model capability + size + context data

Public functions are stable; their `_ollama_*` collaborators are exported
under leading-underscore names because tests pin individual branches of
the jq-style projection they implement.

All functions are best-effort: transport errors, non-2xx responses, and
non-JSON / wrong-shape bodies collapse to neutral values (None / [] /
0.0 / False / an "empty descriptor") so callers can iterate without
sprinkling try/except.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from llm_bench.constants import GIB_BYTES, HTTP_DEFAULT_TIMEOUT_SECONDS, LOG_NAMESPACE
from llm_bench.utils.http import http_get_status, http_post_json

log = logging.getLogger(LOG_NAMESPACE)

# ---------------------------------------------------------------------------
# Readiness-side progress snapshot (used by core.readiness)
# ---------------------------------------------------------------------------


def ollama_progress(url: str, *,
                    timeout: int = HTTP_DEFAULT_TIMEOUT_SECONDS,
                    ) -> tuple[dict | None, int, str]:
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


# ---------------------------------------------------------------------------
# /api/tags — installed model catalogue
# ---------------------------------------------------------------------------


def ollama_tags(url: str, *,
                timeout: int = HTTP_DEFAULT_TIMEOUT_SECONDS,
                ) -> tuple[list | None, int, str]:
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
    for entry in models or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("model") or ""
        if name:
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# Descriptor helpers
#
# `ollama_describe_model` and `ollama_supports_thinking` both need to ask the
# daemon "what's the default model right now?", optionally call /api/show,
# and pull a few values out of the response. We factor each step into its
# own helper so the public functions read top-to-bottom and so other
# call-sites (probes, readiness checks, ...) can reuse the pieces a la
# carte without forking the I/O code.
# ---------------------------------------------------------------------------


def _ollama_get_models_list(base: str, path: str, *,
                            timeout: int = HTTP_DEFAULT_TIMEOUT_SECONDS,
                            ) -> list:
    """GET `<base><path>` and return its parsed `models` array.

    Both /api/ps (currently-loaded) and /api/tags (installed) wrap their
    payload as `{"models":[...]}`, so this is the one helper for both.
    Any transport / non-2xx / non-JSON / wrong-shape failure collapses
    to `[]` — callers iterate, so an empty list is the safe default.
    """
    status, body = http_get_status(f"{base}{path}", timeout=timeout)
    if not (200 <= status < 300):
        return []
    try:
        parsed = json.loads(body or "{}")
    except json.JSONDecodeError:
        return []
    models = parsed.get("models") if isinstance(parsed, dict) else None
    return models if isinstance(models, list) else []


def _ollama_first_name(*sources: list) -> str | None:
    """First usable model name across already-fetched models lists, in the
    priority order they're passed in. Mirrors the bash `${MODEL:-...}`
    fallback chain (typically `(ps_models, tags_models)`).
    """
    for src in sources:
        names = ollama_model_names(src)
        if names:
            return names[0]
    return None


def _find_ollama_entry(models: Any, name: str) -> dict | None:
    """Locate the entry for `name` inside an Ollama models[] list.
    Tolerates either `name` or `model` as the identifier key (same
    coalescing as `ollama_model_names`).
    """
    if not isinstance(models, list):
        return None
    for entry in models:
        if not isinstance(entry, dict):
            continue
        if entry.get("name") == name or entry.get("model") == name:
            return entry
    return None


def _ollama_show(base: str, model: str, *,
                 timeout: int = HTTP_DEFAULT_TIMEOUT_SECONDS) -> dict:
    """POST /api/show {model:<name>}. Returns the parsed body, or `{}`
    on any failure (transport / 4xx / non-JSON / wrong shape). The
    caller decides which fields it cares about; missing fields become
    None downstream naturally.
    """
    try:
        body = http_post_json(f"{base}/api/show",
                              {"model": model}, timeout=timeout)
    except Exception as exc:
        log.warning("ollama /api/show %s failed: %s", model, exc)
        return {}
    return body if isinstance(body, dict) else {}


def _bytes_to_gb(n: Any) -> float:
    """Convert a byte count to GiB rounded to 2 decimals. Bad / missing
    values collapse to 0.0 — same shape as the jq `(... // 0)` fallback.
    Preserves sign so jq-style `(size - other)` differences still work.
    """
    try:
        return round(float(n or 0) / GIB_BYTES, 2)
    except (TypeError, ValueError):
        return 0.0


def _ollama_max_context(show: dict) -> int | None:
    """Pull the first `*.context_length` value out of /api/show's
    `model_info` map.

    Ollama prefixes the key with the arch name (e.g. `qwen3.context_length`,
    `gemma3.context_length`), so we don't hard-code the arch — first
    matching suffix wins. Returns None when there is no such key.
    """
    info = show.get("model_info") if isinstance(show, dict) else None
    if not isinstance(info, dict):
        return None
    for key, value in info.items():
        if isinstance(key, str) and key.endswith(".context_length"):
            return value
    return None


def _ollama_processor_split(running: dict | None) -> str:
    """Mirror the jq processor-split string off `/api/ps`'s entry:

        $r == null                  -> "not loaded"
        $r.size == $r.size_vram     -> "100% GPU"
        $r.size_vram == 0           -> "100% CPU"
        otherwise                   -> "X% GPU / Y% CPU"

    Percentages are integer-rounded the same way jq's `round` rounds.
    """
    if not isinstance(running, dict):
        return "not loaded"
    try:
        size = float(running.get("size") or 0)
        vram = float(running.get("size_vram") or 0)
    except (TypeError, ValueError):
        return "not loaded"
    if size <= 0:
        return "not loaded"
    if size == vram:
        return "100% GPU"
    if vram == 0:
        return "100% CPU"
    gpu_pct = round(vram * 100 / size)
    cpu_pct = round((size - vram) * 100 / size)
    return f"{gpu_pct}% GPU / {cpu_pct}% CPU"


def _empty_ollama_descriptor() -> dict:
    """Zero-valued descriptor for the "daemon has no models at all" path.
    Same shape as the populated one so callers can index uniformly.
    """
    return {
        "model": None,
        "family": None,
        "parameter_size": None,
        "quantization": None,
        "max_context": None,
        "runtime_context": None,
        "disk_gb": 0.0,
        "total_gb": 0.0,
        "vram_gb": 0.0,
        "ram_gb": 0.0,
        "kvcache_gb": 0.0,
        "processor": "not loaded",
        "loaded": False,
    }


def _build_ollama_descriptor(model_name: str, show: dict,
                             running: dict | None,
                             ondisk: dict | None) -> dict:
    """Assemble the final descriptor dict from already-fetched components.
    Pure function: no I/O, no logging — just the jq projection in Python.

    Size math follows jq's `($r.size // 0)` convention: missing sources
    become 0, differences are signed (kvcache_gb on an unloaded model
    will be the negative of disk size, just like the jq does).
    """
    details = show.get("details") if isinstance(show, dict) else None
    if not isinstance(details, dict):
        details = {}

    r_size = (running.get("size") if isinstance(running, dict) else 0) or 0
    r_vram = (running.get("size_vram")
              if isinstance(running, dict) else 0) or 0
    d_size = (ondisk.get("size") if isinstance(ondisk, dict) else 0) or 0

    runtime_context = (running.get("context_length")
                       if isinstance(running, dict) else None)

    return {
        "model": model_name,
        "family": details.get("family"),
        "parameter_size": details.get("parameter_size"),
        "quantization": details.get("quantization_level"),
        "max_context": _ollama_max_context(show),
        "runtime_context": runtime_context,
        "disk_gb": _bytes_to_gb(d_size),
        "total_gb": _bytes_to_gb(r_size),
        "vram_gb": _bytes_to_gb(r_vram),
        "ram_gb": _bytes_to_gb(r_size - r_vram),
        "kvcache_gb": _bytes_to_gb(r_size - d_size),
        "processor": _ollama_processor_split(running),
        "loaded": isinstance(running, dict),
    }


def ollama_describe_model(url: str, *,
                          timeout: int = HTTP_DEFAULT_TIMEOUT_SECONDS) -> dict:
    """Merge /api/ps + /api/tags + /api/show into a single descriptor dict
    for the Ollama daemon's currently-loaded (or first-installed) model.

    Python equivalent of:

        HOST=<url>
        PS=$(curl -s $HOST/api/ps)
        TAGS=$(curl -s $HOST/api/tags)
        MODEL=$(echo "$PS"   | jq -r '.models[0].name // empty')
        MODEL=${MODEL:-$(echo "$TAGS" | jq -r '.models[0].name // empty')}
        curl -s $HOST/api/show -d "{\\"model\\":\\"$MODEL\\"}" | jq '...'

    Field map (jq -> python):
      * .details.family               -> family
      * .details.parameter_size       -> parameter_size
      * .details.quantization_level   -> quantization
      * .model_info[*.context_length] -> max_context (first matching key)
      * $r.context_length             -> runtime_context
      * $d.size / 1 GiB               -> disk_gb     (2-decimal round)
      * $r.size / 1 GiB               -> total_gb
      * $r.size_vram / 1 GiB          -> vram_gb
      * ($r.size - $r.size_vram)/GiB  -> ram_gb
      * ($r.size - $d.size)/GiB       -> kvcache_gb
      * jq processor branch           -> processor
      * $r != null                    -> loaded

    The returned dict is logged at INFO before being handed back. Any
    fetch / parse failure degrades individual fields to None/0.0 rather
    than raising — callers asked for a JSON-ish snapshot, not an
    exception. When no model can be found at all we still return the
    full shape via `_empty_ollama_descriptor()` so downstream code can
    treat the result uniformly.
    """
    base = url.rstrip("/")
    ps_models = _ollama_get_models_list(base, "/api/ps", timeout=timeout)
    tags_models = _ollama_get_models_list(base, "/api/tags", timeout=timeout)

    model_name = _ollama_first_name(ps_models, tags_models)
    if not model_name:
        result = _empty_ollama_descriptor()
        log.info("ollama describe %s: no model found; result=%s",
                 url, json.dumps(result, ensure_ascii=False))
        return result

    running = _find_ollama_entry(ps_models, model_name)
    ondisk = _find_ollama_entry(tags_models, model_name)
    show = _ollama_show(base, model_name, timeout=timeout)

    result = _build_ollama_descriptor(model_name, show, running, ondisk)
    log.info("ollama describe %s: %s",
             url, json.dumps(result, ensure_ascii=False))
    return result


def ollama_supports_thinking(url: str, *,
                             timeout: int = HTTP_DEFAULT_TIMEOUT_SECONDS,
                             ) -> bool:
    """Return True iff the Ollama daemon at `url` exposes a model whose
    /api/show declares the `thinking` capability.

    Python equivalent of:
        HOST=<url>
        MODEL=$(curl -s $HOST/api/ps   | jq -r '.models[0].name // empty')
        MODEL=${MODEL:-$(curl -s $HOST/api/tags | jq -r '.models[0].name // empty')}
        curl -s $HOST/api/show -d "{\\"model\\":\\"$MODEL\\"}" \\
            | jq '.capabilities | index("thinking") != null'

    Any failure along the way (daemon unreachable, no models at all,
    /api/show 4xx, body without a `capabilities` array, ...) collapses
    to False — callers asked for a bool, not an exception, and "we
    couldn't prove it supports thinking" is the safe default.
    """
    base = url.rstrip("/")
    model_name = _ollama_first_name(
        _ollama_get_models_list(base, "/api/ps", timeout=timeout),
        _ollama_get_models_list(base, "/api/tags", timeout=timeout),
    )
    if not model_name:
        return False
    show = _ollama_show(base, model_name, timeout=timeout)
    caps = show.get("capabilities") if isinstance(show, dict) else None
    if not isinstance(caps, list):
        return False
    return "thinking" in caps


__all__ = [
    "ollama_describe_model",
    "ollama_model_names",
    "ollama_progress",
    "ollama_supports_thinking",
    "ollama_tags",
]
