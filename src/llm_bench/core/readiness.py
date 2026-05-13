"""Bundle-based readiness gate.

The two backends have COMPLETELY INDEPENDENT readiness flows because
their wire protocols differ:

  ollama -> GET <entrance>/api/progress   (plain-JSON snapshot, no jobId,
                                            no /cfg) returning
              {app_url, status, model_name, progress, speed_bps,
               duration, completed, completed_at, timestamp, total}
              success: status in {completed, success}
              failure: status in {error, unavailable}
            GET <entrance>/health
              until ANY HTTP 2xx response (body is NOT inspected)

  vllm   -> GET <entrance>/cfg
              -> {jobId,...}                       (single-job)
                 OR
                 {tasks:[{jobId,repo,file,...}], jobIds[], probeUrl,
                  probeIntervalMs}                 (legacy multi-task)
            map config's model_name -> ONE task -> jobId
            GET <entrance>/progress?id=<jobId>     (SSE: `data: {json}`)
              success: status == done
              failure: status == error
            GET <entrance><probeUrl, default /ping>
              until ANY HTTP 2xx response (body is NOT inspected)
            GET <entrance>/v1/models   one-shot for served-name discovery

Both pollers tolerate transport / 5xx / unparseable bodies and retry
every `_FAILURE_RETRY_SECONDS` (5s). vLLM's happy-path poll interval is
the server-supplied `probeIntervalMs` (falling back to the configured
`readiness_probe_interval_seconds` when the server omits it); ollama
uses `readiness_probe_interval_seconds` directly because it has no /cfg
to read.

There is **no outer deadline** — pollers run indefinitely until they
hit a terminal `success` / `error` / `unavailable` status from the
server. The legacy ``api_ready_timeout_minutes`` config field has been
removed; if you need to cap wall-clock time, wrap the run in your own
scheduler (cron / systemd timer with a `RuntimeMaxSec=`).
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable

from llm_bench.clients.ollama_client import ollama_progress
from llm_bench.clients.vllm_client import bundle_cfg, vllm_progress
from llm_bench.constants import LOG_NAMESPACE, READINESS_FAILURE_RETRY_SECONDS
from llm_bench.utils.format import human_bytes
from llm_bench.utils.http import http_get_status

log = logging.getLogger(LOG_NAMESPACE)

# Private alias kept so internal call-sites can stay terse (and so the
# pre-refactor name is still searchable). Source of truth lives in
# llm_bench.constants — see that module for rationale on the value.
_FAILURE_RETRY_SECONDS = READINESS_FAILURE_RETRY_SECONDS


# ----------------------------------------------------------------------
# Ollama bundle vocabulary
# ----------------------------------------------------------------------

_OLLAMA_STATUS_DESCRIPTIONS = {
    "starting": "internal — initial state",
    "waiting": "waiting for ollama to come up",
    "checking": "model already present, checking for an update",
    "pulling manifest": "ollama: fetching manifest",
    "pulling": "ollama: streaming layer bytes",
    "downloading": "bytes transferring (HF direct mode, or wrapper-set)",
    "verifying": "post-download verification",
    "verifying sha256 digest": "post-download verification",
    "writing manifest": "ollama: finishing up (writing manifest)",
    "removing any unused layers": "ollama: finishing up (gc)",
    "hashing": "computing SHA-256 of a local GGUF (GGUF mode)",
    "pushing_blob": "uploading GGUF to ollama (GGUF mode)",
    "blob_pushed": "uploading GGUF to ollama (GGUF mode)",
    "creating": "POST /api/create in progress (GGUF mode)",
    "completed": "model is ready",
    "success": "model is ready",
    "unavailable": "ollama unreachable, or model deleted post-success",
    "error": "failure",
}

# Order matters: more-specific prefixes first so "pulling manifest" doesn't
# get swallowed by "pulling sha256:...".
_OLLAMA_STATUS_PREFIXES = (
    "pulling manifest",
    "writing manifest",
    "removing any unused layers",
    "verifying sha256",
    "pulling",
    "verifying",
)


def _describe_ollama_status(status: str) -> str:
    s = (status or "").strip().lower()
    if not s:
        return ""
    if s in _OLLAMA_STATUS_DESCRIPTIONS:
        return _OLLAMA_STATUS_DESCRIPTIONS[s]
    for prefix in _OLLAMA_STATUS_PREFIXES:
        if s.startswith(prefix):
            return _OLLAMA_STATUS_DESCRIPTIONS.get(
                prefix, _OLLAMA_STATUS_DESCRIPTIONS.get(
                    prefix.split()[0], ""))
    return _OLLAMA_STATUS_DESCRIPTIONS.get(s.split()[0], "")


# ----------------------------------------------------------------------
# vLLM bundle vocabulary
# ----------------------------------------------------------------------

_VLLM_STATUS_DESCRIPTIONS = {
    "queued": "queued, waiting for downloader to pick up",
    "running": "download in progress",
    "done": "download finished",
    "error": "download failed",
}


def _describe_vllm_status(status: str) -> str:
    return _VLLM_STATUS_DESCRIPTIONS.get(
        (status or "").strip().lower(), "")


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------

def wait_until_api_ready(url: str, api_type: str, model: str,
                         *, probe_interval_seconds: float) -> str | None:
    """Block until the bundle finishes download AND the server loads it.

    No outer deadline — pollers run until the server reports a terminal
    status. ``probe_interval_seconds`` is the happy-path cadence used by
    ollama directly; vllm prefers the server-supplied
    ``cfg.probeIntervalMs`` and falls back to this value when missing.

    Returns the server-reported model name when discoverable, else None.
    Source: ollama -> None (no /cfg to read); vllm -> /v1/models
    data[0].id queried after /ping is ok.
    """
    base = url.rstrip("/")
    if api_type == "ollama":
        return _wait_until_ollama_ready(
            base, model, interval=float(probe_interval_seconds))
    return _wait_until_vllm_ready(
        base, model, fallback_interval=float(probe_interval_seconds))


# ----------------------------------------------------------------------
# Ollama branch
# ----------------------------------------------------------------------

def _wait_until_ollama_ready(base: str, model: str, *,
                             interval: float) -> str | None:
    """Wait until ollama finishes pulling the model AND /health = ok.

    No /cfg step — ollama serves a single bundle at a time and exposes
    `/progress` (plain-JSON current snapshot) plus `/health` directly.
    Returns None: the configured `model_name` is used as-is (no
    server-reported name to discover).
    """
    log.info("ollama: poll %s/api/progress (every %.1fs) then %s/health "
             "(every %.1fs); no outer deadline",
             base, interval, base, interval)
    _poll_ollama_progress(base, interval=interval)
    _poll_ollama_health(base, None, interval=interval)
    return None


# ----------------------------------------------------------------------
# vLLM branch
# ----------------------------------------------------------------------

def _wait_until_vllm_ready(base: str, model: str, *,
                           fallback_interval: float) -> str | None:
    log.info("vllm: resolving bundle config at %s/cfg "
             "(no outer deadline)", base)
    cfg = _resolve_bundle_cfg(
        base,
        validator=_vllm_cfg_valid,
        label="vllm bundle",
    )

    task = _find_vllm_task_for_model(cfg, model)
    job_id = str(task.get("jobId") or "").strip()
    if not job_id:
        raise RuntimeError(
            f"vllm bundle /cfg: matched task has no jobId; task={task!r}")
    repo = task.get("repo") or "?"
    ref = task.get("ref") or "?"
    file_name = task.get("file") or "?"
    probe_url = cfg.get("probeUrl") or "/ping"
    interval = _cfg_interval_seconds(cfg.get("probeIntervalMs"),
                                     fallback=fallback_interval)

    log.info("vllm bundle: job=%s repo=%s ref=%s file=%s probe=%s "
             "interval=%.1fs",
             job_id, repo, ref, file_name, probe_url, interval)

    _poll_vllm_progress(base, job_id, interval=interval)
    _poll_vllm_ping(base, probe_url, interval=interval)
    discovered = _discover_vllm_served_name(base)
    if discovered:
        log.info("vllm bundle: discovered served name=%r via /v1/models",
                 discovered)
    return discovered


def _vllm_cfg_valid(cfg: dict) -> bool:
    """Accept any cfg shape that exposes at least one jobId.

    The chart can serve either a single-job descriptor (top-level
    `jobId`, identical to the ollama shape) or a multi-task descriptor
    (`tasks:[{jobId,...},...]` / `jobIds:[...]`). Either is fine —
    `_find_vllm_task_for_model` below picks ONE.
    """
    if str(cfg.get("jobId") or "").strip():
        return True
    tasks = cfg.get("tasks")
    if isinstance(tasks, list) and any(
            str((t or {}).get("jobId") or "").strip() for t in tasks):
        return True
    job_ids = cfg.get("jobIds")
    if isinstance(job_ids, list) and any(
            str(j or "").strip() for j in job_ids):
        return True
    return False


def _find_vllm_task_for_model(cfg: dict, model: str) -> dict:
    """Pick ONE job descriptor out of /cfg.

    Resolution order:
      1) top-level `jobId` (single-job, ollama-style descriptor) — use as-is.
      2) `tasks:[...]` — match config's `model_name` to one task; prio
         exact -> case-insensitive -> substring against task.repo,
         task.file, task.jobId, "<repo>/<file>". Single-task `tasks`
         picks itself regardless of model_name.
      3) `jobIds:[...]` (no `tasks`) — use the first jobId blindly.
    """
    top_job_id = str(cfg.get("jobId") or "").strip()
    if top_job_id:
        return {
            "jobId": top_job_id,
            "repo": cfg.get("repo"),
            "ref": cfg.get("ref"),
            "file": cfg.get("file"),
        }

    tasks = cfg.get("tasks") or []
    if not isinstance(tasks, list) or not tasks:
        job_ids = cfg.get("jobIds") or []
        if isinstance(job_ids, list) and job_ids:
            log.warning("vllm bundle /cfg has jobIds but no tasks "
                        "descriptor; using the first jobId %r blindly",
                        job_ids[0])
            return {"jobId": job_ids[0]}
        cfg_json = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
        raise RuntimeError(
            f"vllm bundle /cfg returned no jobId/tasks/jobIds "
            f"(model={model!r}); cfg={cfg_json}")

    if len(tasks) == 1:
        return tasks[0]

    m = (model or "").strip()
    if not m:
        descr = [(t.get("jobId"), t.get("repo"), t.get("file"))
                 for t in tasks]
        raise RuntimeError(
            "vllm bundle /cfg has multiple tasks but no model_name is "
            f"configured; cannot pick a jobId. tasks={descr}")

    def keys_for(t: dict) -> list:
        repo = (t.get("repo") or "").strip()
        f = (t.get("file") or "").strip()
        jid = (t.get("jobId") or "").strip()
        ks = []
        if jid:
            ks.append(jid)
        if repo:
            ks.append(repo)
        if f:
            ks.append(f)
        if repo and f:
            ks.append(f"{repo}/{f}")
        return ks

    m_lower = m.lower()
    exact: list = []
    case_insensitive: list = []
    partial: list = []
    for t in tasks:
        ks = keys_for(t)
        if any(k == m for k in ks):
            exact.append(t)
        elif any(k.lower() == m_lower for k in ks):
            case_insensitive.append(t)
        elif any((m_lower in k.lower()) or (k.lower() in m_lower)
                 for k in ks if k):
            partial.append(t)

    for bucket, kind in ((exact, "exact"),
                         (case_insensitive, "case-insensitive"),
                         (partial, "partial")):
        if bucket:
            chosen = bucket[0]
            if len(bucket) > 1:
                log.warning("vllm bundle /cfg: model_name=%r matched %d "
                            "tasks via %s (%s); using first",
                            model, len(bucket), kind,
                            [t.get("jobId") for t in bucket])
            elif kind != "exact":
                log.info("vllm bundle /cfg: model_name=%r resolved by "
                         "%s match to task jobId=%s repo=%s file=%s",
                         model, kind, chosen.get("jobId"),
                         chosen.get("repo"), chosen.get("file"))
            return chosen

    descr = [(t.get("jobId"), t.get("repo"), t.get("file")) for t in tasks]
    raise RuntimeError(
        f"vllm bundle /cfg: model_name={model!r} doesn't match any "
        f"task (repo/file/jobId). tasks={descr}")


def _discover_vllm_served_name(base: str) -> str | None:
    """One-shot GET /v1/models -> data[0].id. Best-effort; None on error."""
    try:
        status, body = http_get_status(f"{base}/v1/models", timeout=10)
        if not (200 <= status < 300):
            return None
        items = json.loads(body or "{}").get("data") or []
    except Exception:
        return None
    ids = [i for i in ((it or {}).get("id") for it in items) if i]
    return ids[0] if ids else None


# ----------------------------------------------------------------------
# Shared core: cfg / progress / probe pollers
# ----------------------------------------------------------------------

def _cfg_interval_seconds(probe_interval_ms,
                          *, fallback: float) -> float:
    """Convert /cfg's `probeIntervalMs` to seconds; clamp to >=1s, fall
    back to ``fallback`` (the configured
    ``readiness_probe_interval_seconds``) when missing / unparseable /
    zero. The previous default was the hard-coded
    ``_FAILURE_RETRY_SECONDS``; surfacing it via config lets users
    speed up / slow down the steady-state cadence without recompiling.
    """
    try:
        ms = int(probe_interval_ms)
    except (TypeError, ValueError):
        ms = 0
    if ms > 0:
        return max(1.0, ms / 1000.0)
    return float(fallback)


def _truncate(text: str, limit: int = 500) -> str:
    """Trim a raw HTTP body for log output; keeps log lines manageable
    while still showing what the server actually returned.
    """
    s = (text or "").strip()
    if not s:
        return "(empty)"
    return s if len(s) <= limit else (s[:limit] + f"... [+{len(s)-limit} chars]")


def _resolve_bundle_cfg(base: str, *,
                        validator: Callable[[dict], bool],
                        label: str) -> dict:
    """GET <base>/cfg until validator(data)==True. Retries every
    `_FAILURE_RETRY_SECONDS` on transport failure / non-2xx / invalid
    body so the chart launcher gets time to come up. No outer deadline;
    runs until the chart launcher serves a usable /cfg.
    """
    attempt = 0
    while True:
        attempt += 1
        data, status, raw = bundle_cfg(base)
        if data is not None:
            body_json = json.dumps(data, sort_keys=True, ensure_ascii=False)
            if validator(data):
                log.info("%s: /cfg resolved after %d probe(s) (HTTP %s): %s",
                         label, attempt, status, body_json)
                return data
            last_msg = (f"/cfg HTTP {status} body has no usable jobId yet: "
                        f"{body_json}")
        else:
            last_msg = (f"/cfg HTTP {status} or non-JSON body "
                        f"(chart launcher still spinning up?): "
                        f"{_truncate(raw)}")
        log.info("%s: %s; sleeping %ds before retry",
                 label, last_msg, _FAILURE_RETRY_SECONDS)
        time.sleep(_FAILURE_RETRY_SECONDS)


def _format_vllm_in_flight(body: dict, status_text: str, err: str) -> str:
    """vllm /progress SSE event shape:
        {id, status, err, isBundle, total, downloaded, speed_bps,
         human_total, human_done, human_spd, repo, ref}
    """
    total = body.get("total")
    downloaded = body.get("downloaded")
    if total and isinstance(total, (int, float)) and total > 0:
        pct = (float(downloaded or 0) / float(total)) * 100
        return (f"{status_text or '?'} — "
                f"{body.get('human_done') or human_bytes(downloaded)}"
                f" / "
                f"{body.get('human_total') or human_bytes(total)} "
                f"({pct:.1f}%) @ "
                f"{body.get('human_spd') or '?'}")
    msg = status_text or "?"
    if err:
        msg = f"{msg} — err={err!r}"
    return msg


def _format_ollama_in_flight(body: dict, status_text: str, err: str) -> str:
    """ollama /progress snapshot shape:
        {app_url, status, model_name, progress (0-100), speed_bps,
         duration (s), completed, completed_at, timestamp, total}

    Different from the vllm shape — there's no human_* pre-formatted
    string, the progress is a percentage rather than byte counts, and
    `total` / `completed` may be 0 for completed runs (some upstreams
    don't expose byte totals).
    """
    parts = []
    model_name = (body.get("model_name") or "").strip()
    if model_name:
        parts.append(f"model={model_name}")
    progress = body.get("progress")
    if isinstance(progress, (int, float)):
        parts.append(f"progress={float(progress):.1f}%")
    speed_bps = body.get("speed_bps")
    if isinstance(speed_bps, (int, float)) and speed_bps > 0:
        parts.append(f"speed={human_bytes(speed_bps)}/s")
    duration = body.get("duration")
    if isinstance(duration, (int, float)) and duration > 0:
        parts.append(f"duration={int(duration)}s")

    head = status_text or "?"
    if parts:
        tail = ", ".join(parts)
        return f"{head} — {tail}" + (f" — err={err!r}" if err else "")
    if err:
        return f"{head} — err={err!r}"
    return head


def _log_progress_parse_failure(label: str, endpoint: str,
                                http_status: int, raw: str,
                                attempt: int) -> None:
    """Pure logging helper used by both pollers when the body can't be
    parsed (transport error, non-2xx, non-JSON, no SSE events, ...).

    `endpoint` is the path portion (e.g. "/api/progress" or "/progress")
    so the warning matches the actual URL the poller hit — useful when
    triaging routing / auth-redirect issues against the chart.
    """
    log.warning("%s: transport/HTTP %s or unparseable body from "
                "%s: %s (attempt %d); sleeping %ds",
                label, http_status, endpoint, _truncate(raw), attempt,
                _FAILURE_RETRY_SECONDS)


# ----------------------------------------------------------------------
# Ollama progress poller (plain-JSON snapshot, ollama-only states)
# ----------------------------------------------------------------------

def _poll_ollama_progress(base: str, *, interval: float) -> None:
    """Poll ollama's plain-JSON /progress (no query string) until
    terminal status. No outer deadline — runs until the server reports
    a terminal status.

    Terminal states (per the ollama bundle spec):
      success: completed | success
      failure: error | unavailable

    All other statuses (starting / waiting / checking / pulling manifest
    / pulling / downloading / verifying / writing manifest / hashing /
    pushing_blob / creating / ...) are in-flight: log and re-poll every
    `interval`. Transport / non-JSON failures retry every
    `_FAILURE_RETRY_SECONDS`.
    """
    log.info("ollama: polling %s/api/progress every %.1fs "
             "(fallback %ds on transport / non-JSON; no outer deadline)",
             base, interval, _FAILURE_RETRY_SECONDS)
    attempt = 0
    while True:
        attempt += 1
        body, http_status, raw = ollama_progress(base)
        if body is None:
            _log_progress_parse_failure("ollama progress", "/api/progress",
                                        http_status, raw, attempt)
            time.sleep(_FAILURE_RETRY_SECONDS)
            continue

        status_text = str(body.get("status") or "").strip()
        status_key = status_text.lower()
        err = str(body.get("err") or "").strip()
        body_json = json.dumps(body, sort_keys=True, ensure_ascii=False)

        if status_key in ("completed", "success"):
            summary = _format_ollama_in_flight(body, status_text, err)
            log.info("ollama progress: %s — download finished  body=%s",
                     summary, body_json)
            return

        if status_key == "error":
            raise RuntimeError(
                f"ollama /api/progress reported status=error "
                f"(message={err or '(no error_message)'}); body={body_json}")

        if status_key == "unavailable":
            raise RuntimeError(
                f"ollama /api/progress reported status=unavailable "
                f"(ollama unreachable, or model deleted post-success; "
                f"message={err or '(no error_message)'}); body={body_json}")

        last_msg = _format_ollama_in_flight(body, status_text, err)
        description = _describe_ollama_status(status_text)
        if description:
            log.info("ollama progress: %s  [%s]  body=%s",
                     last_msg, description, body_json)
        else:
            log.info("ollama progress: %s  body=%s",
                     last_msg, body_json)
        time.sleep(interval)


# ----------------------------------------------------------------------
# vLLM progress poller (SSE `data: {...}` framing, vllm-only states)
# ----------------------------------------------------------------------

def _poll_vllm_progress(base: str, job_id: str, *,
                        interval: float) -> None:
    """Poll vllm's SSE /progress until terminal status. No outer
    deadline — runs until the server reports a terminal status.

    Wire format: each request returns one or more SSE events of the form
    `data: {"id":"...","status":"<state>",...}`. `vllm_progress()` picks
    the LATEST event from the body and returns its parsed dict.

    Terminal states (per the vllm bundle spec):
      success: status == done
      failure: status == error

    queued / running are in-flight: log and re-poll using the
    cfg-supplied `interval`. Transport / non-parseable bodies retry
    every `_FAILURE_RETRY_SECONDS`.
    """
    log.info("vllm bundle: polling %s/progress?id=%s every %.1fs "
             "(SSE; fallback %ds on transport / non-parseable; "
             "no outer deadline)",
             base, job_id, interval, _FAILURE_RETRY_SECONDS)
    attempt = 0
    while True:
        attempt += 1
        body, http_status, raw = vllm_progress(base, job_id)
        if body is None:
            _log_progress_parse_failure("vllm bundle progress",
                                        f"/progress?id={job_id}",
                                        http_status, raw, attempt)
            time.sleep(_FAILURE_RETRY_SECONDS)
            continue

        status_text = str(body.get("status") or "").strip()
        status_key = status_text.lower()
        err = str(body.get("err") or "").strip()
        body_json = json.dumps(body, sort_keys=True, ensure_ascii=False)

        if status_key == "done":
            done = body.get("human_done") or human_bytes(
                body.get("downloaded"))
            total = body.get("human_total") or human_bytes(
                body.get("total"))
            log.info("vllm bundle progress: done — download finished "
                     "(%s / %s)  body=%s", done, total, body_json)
            return

        if status_key == "error":
            raise RuntimeError(
                f"vllm bundle {job_id!r} reported status=error "
                f"(message={err or '(no error_message)'}); body={body_json}")

        last_msg = _format_vllm_in_flight(body, status_text, err)
        description = _describe_vllm_status(status_text)
        if description:
            log.info("vllm bundle progress: %s  [%s]  body=%s",
                     last_msg, description, body_json)
        else:
            log.info("vllm bundle progress: %s  body=%s",
                     last_msg, body_json)
        time.sleep(interval)


# ----------------------------------------------------------------------
# Ollama health poller (looks at the JSON body)
# ----------------------------------------------------------------------

def _poll_ollama_health(base: str, probe_url: str | None, *,
                        interval: float) -> None:
    """Poll ollama's /health until ANY HTTP 2xx response. No outer
    deadline — runs until /health answers 2xx.

    The chart's /health endpoint starts answering 2xx after the model
    is loaded into ollama; the body is NOT inspected (response can be
    plain text, empty, or any JSON shape). Anything else — transport
    error, 404 (route not wired yet), 503 (still loading), HTML auth
    redirect — is logged with a truncated raw body and retried.
    `probe_url` is normally None (defaults to '/health' on `base`); a
    non-empty value lets tests / future callers override.
    """
    full_url = _build_probe_url(base, probe_url, "/health")
    log.info("ollama: waiting for model load at %s every %.1fs "
             "(HTTP 2xx = ready; no outer deadline)", full_url, interval)
    attempt = 0
    while True:
        attempt += 1
        http_status, raw = http_get_status(full_url)
        if 200 <= http_status < 300:
            log.info("ollama health: HTTP %s after %d probe(s) — "
                     "model loaded  body=%s",
                     http_status, attempt, _truncate(raw, 300))
            return
        if http_status == 0:
            log.warning("ollama health: %s transport error "
                        "(connection refused / TLS / timeout)", full_url)
        else:
            log.info("ollama health: HTTP %s still loading: %s",
                     http_status, _truncate(raw, 200))
        time.sleep(interval)


# ----------------------------------------------------------------------
# vLLM ping poller (HTTP 2xx is the ONLY thing we look at)
# ----------------------------------------------------------------------

def _poll_vllm_ping(base: str, probe_url: str, *,
                    interval: float) -> None:
    """Poll vllm's /ping until ANY HTTP 2xx response. No outer deadline
    — runs until /ping answers 2xx.

    The vllm-side chart's /ping starts answering 2xx after the backend
    has finished bringing the model online; the body is NOT inspected
    (it may be `pong`, `{}`, `{"status":"ok"}`, or absent). Anything
    else — transport error, 404 (route not wired yet), 503 (still
    loading) — is logged and retried indefinitely.
    `probe_url` may be relative ('/ping') or absolute; empty falls back
    to '/ping' on `base`.
    """
    full_url = _build_probe_url(base, probe_url, "/ping")
    log.info("vllm bundle: waiting for model load at %s every %.1fs "
             "(HTTP 2xx = ready; no outer deadline)", full_url, interval)
    attempt = 0
    while True:
        attempt += 1
        http_status, raw = http_get_status(full_url)
        if 200 <= http_status < 300:
            log.info("vllm bundle health: HTTP %s after %d probe(s) — "
                     "model loaded", http_status, attempt)
            return
        if http_status == 0:
            log.warning("vllm bundle health: %s transport error "
                        "(connection refused / TLS / timeout)", full_url)
        else:
            log.info("vllm bundle health: HTTP %s still loading: %s",
                     http_status, _truncate(raw, 200))
        time.sleep(interval)


def _build_probe_url(base: str, probe_url: str | None,
                     default_path: str) -> str:
    p = (probe_url or "").strip() or default_path
    if p.startswith(("http://", "https://")):
        return p
    if not p.startswith("/"):
        p = "/" + p
    return f"{base.rstrip('/')}{p}"
