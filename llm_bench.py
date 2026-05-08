#!/usr/bin/env python3
"""Sequential LLM benchmark for Olares-hosted Ollama / vLLM charts.

For every entry in `models`, the script:
  1. checks whether the chart is already installed
       - state=running                : reuse it, skip install
       - state=installing/pending/... : `market status --watch` to wait
       - state=failed/stopped/...     : uninstall (with delete-data)
                                        then reinstall
       - not installed                : `market install --watch`
  2. discovers the entrance URL via `olares-cli settings apps get`
     (the dedicated `entrances list` endpoint returns spec.entrances
     raw with url="", so it can't be the primary source). For
     authLevel=internal entrances that have no public domain the
     script falls back to a cluster-internal `<svc>.<ns>:<port>` URL
     reconstructed from `apps get`'s ports[]. A per-model
     `endpoint_url` in the config overrides everything.
  3. (optional) pulls weights via /api/pull when api_type=ollama
  4. warms the model up, then runs every prompt in `questions` with
     stream=false and records timing.

     For api_type=ollama the per-prompt record uses the precise
     server-side fields from `/api/generate` (load + prompt_eval +
     eval). For api_type=openai (vLLM / llama.cpp / other oai-compat
     backends) the record additionally:
       - issues a separate max_tokens=1 round-trip to APPROXIMATE TTFT
         (true TTFT would need stream=true);
       - reads `usage.completion_tokens|prompt_tokens|total_tokens` and
         falls back to a char-count estimate when usage is missing;
       - reads llama.cpp's `timings.predicted_ms / predicted_per_second`
         when present so server-side decode TPS is reported alongside the
         client end-to-end TPS;
       - honors per-model knobs (max_tokens / temperature / top_p /
         api_key / extra_headers / extra_body / endpoint=chat|completion)
         configurable globally under `openai_defaults` and per-model
         under `openai`.
  5. uninstalls (with --delete-data by default) so the GPU memory
     and disk are freed before the next model.
     Two ways to keep the model around:
       - `uninstall_after_run: false` (global or per-model) skips the
         post-benchmark uninstall unconditionally.
       - `preserve_if_existed: true` skips it only when the model was
         already installed before this run started.

By default the script auto-flips an installed entrance to authLevel=public
right after find_entrance — this is needed because most LLM charts ship
authLevel=internal and the script has no Olares user cookie/JWT, so any
HTTP request would hit the Authelia login page. The setting is per-app
and dies with the app on uninstall, so it's not persistent. Switch off
with `auto_open_internal_entrance: false` (global or per-model) if you'd
rather flip it yourself; the script will then error out with the exact
manual command instead of failing on a silent 302/401. Setting the legacy
flag `set_public_during_run: true` is treated as a synonym.

User-level env variables (e.g. `OLARES_USER_HUGGINGFACE_TOKEN` for the
vLLM charts) are NOT managed by this script anymore — set them once
manually via `olares-cli settings advanced env user set --var KEY=VAL`
and they will be picked up by every chart that references them via
`valueFrom`. The optional `user_envs` block in config (legacy) still
works as an escape hatch but is no longer documented in the example.

After all models finish, an HTML + JSON report is written to disk and
a single email (HTML body + raw JSON attachment) is sent via SMTP.

The path to the olares-cli binary is configurable via `--cli-path`
or the config field `cli_path` (default: look up `olares-cli` on
PATH). This matters when the binary lives in an unusual location
(e.g. `/home/olares/test/olares-cli`).

Only depends on the Python 3.9+ stdlib so it can run on any Olares
machine without `pip install`. Designed to be triggered from cron;
see config.example.json for the configuration shape.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import smtplib
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape as html_escape
from typing import Any, Optional

log = logging.getLogger("llm_bench")


# Set by setup_cli_path() at startup. Defaults to whatever the OS resolves
# from PATH, but `--cli-path` / config `cli_path` can override it (useful
# when olares-cli isn't installed system-wide, e.g. /home/olares/test/).
_CLI_PATH = "olares-cli"


def set_cli_path(path: str) -> None:
    """Override the olares-cli binary used by every helper."""
    global _CLI_PATH
    if path:
        _CLI_PATH = path
        log.info("using olares-cli at %s", _CLI_PATH)


def cli() -> str:
    return _CLI_PATH


# --------------------------------------------------------------------------- #
# Subprocess wrappers around olares-cli
# --------------------------------------------------------------------------- #

def _run(cmd: list[str], *, timeout: int, capture: bool = False,
         check: bool = True) -> subprocess.CompletedProcess:
    log.info("$ %s", " ".join(cmd))
    return subprocess.run(cmd, timeout=timeout, capture_output=capture,
                          text=True, check=check)


def cli_json(args: list[str], *, timeout: int = 60) -> Any:
    """Run `olares-cli ... -o json` and decode stdout."""
    proc = _run([cli(), *args, "-o", "json"],
                timeout=timeout, capture=True)
    out = proc.stdout.strip()
    return json.loads(out) if out else None


def apply_user_envs(user_envs: dict[str, str]) -> None:
    """Set user-level env variables once before the install loop.

    `OLARES_USER_HUGGINGFACE_TOKEN` (and similar referenced-by-chart vars)
    is per-USER, not per-app, so we write it once for the whole run via
    `olares-cli settings advanced env user set --var KEY=VALUE`.
    """
    if not user_envs:
        return
    cmd = [cli(), "settings", "advanced", "env", "user", "set"]
    for key, value in user_envs.items():
        cmd.extend(["--var", f"{key}={value}"])
    log.info("applying %d user-level env var(s): %s",
             len(user_envs), ", ".join(user_envs.keys()))
    _run(cmd, timeout=120)


def market_install(app: str, *, watch_minutes: int,
                   envs: Optional[list[str]] = None) -> None:
    """Install via the per-user market API and block until state=running."""
    cmd = [cli(), "market", "install", app,
           "--watch", "--watch-timeout", f"{watch_minutes}m"]
    for kv in envs or []:
        cmd.extend(["--env", kv])
    _run(cmd, timeout=watch_minutes * 60 + 60)


def market_uninstall(app: str, *, watch_minutes: int = 30,
                     delete_data: bool = True, cascade: bool = True) -> None:
    cmd = [cli(), "market", "uninstall", app,
           "--watch", "--watch-timeout", f"{watch_minutes}m"]
    if cascade:
        cmd.append("--cascade")
    if delete_data:
        cmd.append("--delete-data")
    _run(cmd, timeout=watch_minutes * 60 + 60)


# State buckets — keep in sync with cli/cmd/ctl/market/watch.go and
# framework/app-service/pkg/appstate/state_transition.go. We do not
# enumerate every transient (`installingCanceling` etc.); an unknown
# state falls into the "needs reinstall" path which is the safe default.
RUNNING_STATES = {"running"}
PROGRESSING_STATES = {
    "pending", "downloading", "installing", "initializing",
    "upgrading", "applyingEnv", "resuming",
}
RECOVERABLE_STATES = {"stopped", "suspended"}  # handle via uninstall+reinstall

# States where the helm release is already gone (or never landed) and the
# state machine ONLY allows InstallOp — calling `uninstall` here would 400
# with "UninstallOp operation is not allowed for <state> state". Skip the
# doomed pre-uninstall and go straight to install. Source of truth:
# `OperationAllowedInState` in
# framework/app-service/pkg/appstate/state_transition.go.
ROLLED_BACK_STATES = {
    "installingCanceled", "pendingCanceled", "downloadingCanceled",
    "installFailed", "downloadFailed",
}


def get_app_state(app: str) -> Optional[dict]:
    """Returns the current statusRow dict, or None if the app isn't installed.

    Wraps `olares-cli market status <app> -a -o json`. The CLI returns a
    non-zero exit when the app isn't installed (`runStatusSingle`'s
    failOp path), so we explicitly disable check= and treat both the
    failure exit and an empty stdout as "not installed".
    """
    proc = subprocess.run(
        [cli(), "market", "status", app, "-a", "-o", "json"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    if not out:
        return None
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        log.warning("could not decode `market status %s -o json` stdout: %s",
                    app, out[:200])
        return None
    # Single-app status returns a dict; multi-source returns a list.
    if isinstance(data, list):
        data = data[0] if data else None
    return data if isinstance(data, dict) else None


def market_status_watch(app: str, *, watch_minutes: int) -> None:
    """Block until the app reaches a terminal state (op-agnostic).

    Used to wait out an in-flight install/upgrade we found at startup.
    """
    _run(
        [cli(), "market", "status", app,
         "--watch", "--watch-timeout", f"{watch_minutes}m"],
        timeout=watch_minutes * 60 + 60,
    )


def ensure_installed(app: str, *, install_minutes: int,
                     uninstall_minutes: int, install_envs: list[str],
                     delete_data: bool, skip_if_running: bool
                     ) -> tuple[bool, str]:
    """Make sure the app is in a `running` state before benchmarking.

    Returns (already_existed, decision) where decision is one of
    'reused' / 'fresh' / 'recovered' so the caller can label the report
    and decide whether to skip the post-benchmark uninstall.
    """
    row = get_app_state(app)
    if row is None:
        log.info("%s not installed, installing...", app)
        market_install(app, watch_minutes=install_minutes, envs=install_envs)
        return (False, "fresh")

    state = (row.get("state") or "").strip()
    log.info("%s pre-existing state=%r op=%r", app, state, row.get("opType"))

    if state in RUNNING_STATES and skip_if_running:
        log.info("%s already running -> skipping install", app)
        return (True, "reused")

    if state in PROGRESSING_STATES:
        log.info("%s mid-lifecycle (%s); waiting for terminal state...",
                 app, state)
        market_status_watch(app, watch_minutes=install_minutes)
        row = get_app_state(app)
        new_state = (row or {}).get("state", "")
        if new_state in RUNNING_STATES:
            log.info("%s converged to running after watch", app)
            return (True, "reused")
        log.warning("%s landed in %r after watch; reinstalling", app, new_state)
        state = new_state

    if state in ROLLED_BACK_STATES:
        log.info("%s in rolled-back state %r -> reinstalling directly "
                 "(no uninstall needed: helm release already gone)",
                 app, state)
        market_install(app, watch_minutes=install_minutes, envs=install_envs)
        return (False, "recovered")

    # Anything else (stopped / *Failed / unknown) — clean slate to make
    # the benchmark deterministic.
    log.warning("%s in non-running state (%s); uninstall + reinstall", app, state)
    try:
        market_uninstall(app, watch_minutes=uninstall_minutes,
                         delete_data=delete_data)
    except Exception as exc:  # noqa: BLE001
        log.warning("pre-install uninstall failed (continuing): %s", exc)
    market_install(app, watch_minutes=install_minutes, envs=install_envs)
    return (False, "recovered")


def _normalize_url(raw: str) -> str:
    """Add an https:// scheme if the entrance URL came back bare-host."""
    raw = (raw or "").strip().rstrip("/")
    if not raw:
        return ""
    if raw.startswith(("http://", "https://")):
        return raw
    return "https://" + raw


def _cluster_url_from_ports(info: dict) -> Optional[str]:
    """Build an in-cluster URL from `apps get`'s ports[] + namespace.

    Used when the entrance has authLevel=internal (no public domain).
    K3s/kubernetes resolve `<svc>.<ns>` from any pod or — on a typical
    single-node Olares host — from the node itself, so the script can
    still reach the service without changing the auth level.

    Picks the first TCP port. Returns None if there's nothing usable.
    """
    namespace = (info or {}).get("namespace", "")
    ports = (info or {}).get("ports") or []
    for p in ports:
        host = (p or {}).get("host", "")
        port = (p or {}).get("port", 0)
        proto = (p or {}).get("protocol", "tcp") or "tcp"
        if not host or not port or proto.lower() != "tcp":
            continue
        if namespace:
            return f"http://{host}.{namespace}:{port}"
        return f"http://{host}:{port}"
    return None


def find_entrance(app: str, hint: Optional[str],
                  *, override: Optional[str] = None
                  ) -> tuple[str, str, str]:
    """Pick the entrance to hit. Returns (entrance_name, base_url, auth_level).

    auth_level is one of "public" / "private" / "internal" / "" (when an
    explicit override URL is given and we don't know the level).

    Resolution order (each step short-circuits as soon as it succeeds):

    1. Explicit per-model `endpoint_url` override — caller wins; auth_level
       is reported as "" because we never look at the entrance.
    2. `settings apps get <app>` -> entrances[i].url
       (only the /api/myapps path runs GenEntranceURL; the dedicated
       /entrances endpoint returns app.Spec.Entrances raw with url="",
       so we must NOT rely on `apps entrances list` for the URL.)
    3. cluster-internal URL built from ports[] + namespace
       (typical when the chart only ships an authLevel=internal entrance
       AND publishes a service port). Works when the script runs on the
       Olares host or in a pod on the same cluster. auth_level is
       reported as "internal" so callers know it's effectively bypassed.

    The returned auth_level is what the caller needs to decide whether
    to flip the entrance to "public" before sending the benchmark
    request — see `ensure_entrance_public`. internal/private entrances
    fronted by Authelia will reject anonymous HTTP from this script
    even when the entrance.url field is populated.
    """
    if override:
        log.info("%s: using endpoint_url override %s", app, override)
        return (hint or "override", override.rstrip("/"), "")

    info = cli_json(["settings", "apps", "get", app])
    rows = (info or {}).get("entrances") or []
    if not rows:
        # Last-resort fallback: try the dedicated entrances endpoint to at
        # least surface the entrance NAME (URL will be empty there).
        rows = cli_json(["settings", "apps", "entrances", "list", app]) or []
        if not isinstance(rows, list):
            rows = []
    if not rows:
        raise RuntimeError(f"no entrances reported for app {app!r}")

    def _pick(entries: list[dict]) -> dict:
        if hint:
            for r in entries:
                if r.get("name") == hint:
                    return r
            log.warning("entrance hint %r not in %s, falling back",
                        hint, [r.get("name") for r in entries])
        for keyword in ("ollama", "api", "vllm", "openai", "client", "llamacpp"):
            for r in entries:
                if keyword in (r.get("name") or "").lower():
                    return r
        return entries[0]

    runnable = [r for r in rows if r.get("url")]
    if runnable:
        chosen = _pick(runnable)
        return (
            chosen.get("name") or "entrance",
            _normalize_url(chosen["url"]),
            (chosen.get("authLevel") or "").lower(),
        )

    # No public URL on any entrance. Try the cluster-internal fallback.
    if info:
        internal = _cluster_url_from_ports(info)
        if internal:
            chosen = _pick(rows)
            log.warning(
                "%s: no public entrance URL; "
                "falling back to in-cluster service URL %s",
                app, internal,
            )
            # Treated as "internal" because the cluster-DNS URL bypasses
            # Authelia entirely; no auth flip needed.
            return (chosen.get("name") or "entrance", internal, "internal")

    raise RuntimeError(
        f"no reachable entrance for app {app!r}: entrances={rows}, "
        f"namespace={(info or {}).get('namespace')!r}, "
        f"ports={(info or {}).get('ports')}. "
        "Set `endpoint_url` for this model in the config to point at the "
        "API directly (e.g. http://<svc>.<ns>:<port>), or enable "
        "`auto_open_internal_entrance: true` so the script flips the "
        "entrance auth-level to public on every run."
    )


def get_entrance_auth_level(app: str, entrance: str) -> Optional[str]:
    """Re-read the live authLevel of one entrance via `apps get -o json`.

    Returns None if the entrance is gone (or apps get failed); else one
    of "public" / "private" / "internal".
    """
    try:
        info = cli_json(["settings", "apps", "get", app])
    except Exception as exc:  # noqa: BLE001
        log.debug("apps get %s failed: %s", app, exc)
        return None
    for r in (info or {}).get("entrances") or []:
        if r.get("name") == entrance:
            level = (r.get("authLevel") or "").lower()
            return level or None
    return None


def open_entrance(app: str, entrance: str, *, verify_timeout: int = 30,
                  poll_interval: float = 2.0) -> None:
    """Flip an entrance's auth-level + default-policy to "public".

    Required when the script runs without Olares user auth and the
    entrance was installed with authLevel != public (the common case
    for vLLM / llama.cpp charts which ship authLevel=internal). The
    setting is per-app, so it lives only as long as the app does;
    `uninstall_after_run` (default true) removes it together with the
    app, so this is a no-op for the next install of the same chart.

    After the two writes we POLL `apps get` for up to verify_timeout
    seconds because the controller takes a beat to apply the change to
    the ingress; otherwise the immediate next request would still hit
    the old (non-public) policy and 302 to a login page.
    """
    log.info("%s/%s: flipping auth-level + policy to public", app, entrance)
    _run([cli(), "settings", "apps", "auth-level", "set",
          app, entrance, "--level", "public"], timeout=60)
    _run([cli(), "settings", "apps", "policy", "set",
          app, entrance, "--default-policy", "public"], timeout=60)
    deadline = time.monotonic() + verify_timeout
    last = None
    while time.monotonic() < deadline:
        last = get_entrance_auth_level(app, entrance)
        if last == "public":
            log.info("%s/%s: authLevel is now public", app, entrance)
            return
        time.sleep(poll_interval)
    raise RuntimeError(
        f"open_entrance: {app}/{entrance} stayed at authLevel={last!r} "
        f"after {verify_timeout}s; the auth-level write succeeded but the "
        "ingress hasn't picked it up. Re-run, or flip manually:\n"
        f"  olares-cli settings apps auth-level set {app} {entrance} --level public\n"
        f"  olares-cli settings apps policy set {app} {entrance} --default-policy public"
    )


def ensure_entrance_public(app: str, entrance: str, current_level: str,
                           *, auto_open: bool) -> str:
    """Make sure the entrance is `public` so unauthenticated HTTP works.

    Behavior matrix:
      - current_level=="public"            -> no-op, return "public"
      - current_level==""                  -> no-op (override URL; we
        don't manage the entrance), return ""
      - current_level in {private,internal} and auto_open=True
                                           -> call open_entrance,
        return "public"
      - current_level in {private,internal} and auto_open=False
                                           -> raise with manual command
    """
    level = (current_level or "").lower()
    if level == "public" or level == "":
        return level
    if not auto_open:
        raise RuntimeError(
            f"entrance {app}/{entrance} has authLevel={level!r} but "
            "auto_open_internal_entrance=false; the script can't reach the "
            "API anonymously. Either flip the entrance to public:\n"
            f"  olares-cli settings apps auth-level set {app} {entrance} --level public\n"
            f"  olares-cli settings apps policy set {app} {entrance} --default-policy public\n"
            "or set `endpoint_url` in the model config to point at an "
            "in-cluster URL that bypasses the ingress."
        )
    open_entrance(app, entrance)
    return "public"


# --------------------------------------------------------------------------- #
# HTTP helper
# --------------------------------------------------------------------------- #

def http_post_json(url: str, payload: dict, *, timeout: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json",
                 "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get_status(url: str, *, timeout: int = 10) -> tuple[int, str]:
    """GET `url`. Returns (status, body); (0, "") on transport-level failure.

    The readiness probe wants to retry on ANY non-2xx, including
    `ConnectionRefusedError`, `TimeoutError`, ingress 502/503, etc.,
    so we deliberately swallow transport errors here and translate
    them into status=0 for the caller.
    """
    req = urllib.request.Request(
        url, method="GET",
        headers={"Accept": "application/json"})
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


def wait_until_api_ready(
    url: str, api_type: str, model: str,
    *,
    max_wait_minutes: int = 60,
    probe_interval_seconds: int = 30,
) -> None:
    """Block until the in-container LLM API actually accepts requests.

    `market install --watch` only signals that the chart reached
    `state=running`; the LLM server inside the container may still be
    pulling weights from HuggingFace or loading them into VRAM. For
    vLLM / llama.cpp that gap is typically minutes, sometimes tens of
    minutes for very large models, and there's no chart-level signal
    for it. So we just poll the live API.

    Probes:
      ollama  -> GET /api/tags    (daemon liveness; cheap, instant.
                                  The actual model download still
                                  happens later in /api/pull, which
                                  is the caller's responsibility.)
      openai  -> GET /v1/models   (vLLM / llama-server only register
                                  this *after* weights finish loading.
                                  An empty `data=[]` means still loading.)

    Tolerates every transport / 5xx / non-JSON failure and just sleeps
    + retries. Raises only when the wall-clock budget is exhausted.
    """
    deadline = time.monotonic() + max_wait_minutes * 60
    base = url.rstrip("/")
    probe = f"{base}/api/tags" if api_type == "ollama" else f"{base}/v1/models"
    log.info("waiting for API ready: GET %s (max %dm, probe every %ds)",
             probe, max_wait_minutes, probe_interval_seconds)
    attempt = 0
    last_msg = ""
    while True:
        attempt += 1
        status, body = _http_get_status(probe, timeout=10)
        ready = False
        if 200 <= status < 300:
            if api_type == "ollama":
                ready = True
                last_msg = f"HTTP {status}"
            else:
                try:
                    data = json.loads(body or "{}")
                    items = data.get("data") or []
                except json.JSONDecodeError:
                    items = []
                    last_msg = "200 but body not JSON (login page?)"
                if items:
                    ids = [(it or {}).get("id") for it in items]
                    if model and model not in ids:
                        last_msg = (
                            f"HTTP {status}, served ids={ids}, "
                            f"{model!r} not in list (continuing; chart "
                            "probably set served-model-name)")
                    else:
                        last_msg = f"HTTP {status}, ids={ids}"
                    ready = True
                else:
                    last_msg = ("200 but data=[] "
                                "(server up, weights still loading)")
        else:
            last_msg = f"status={status}"
        if ready:
            log.info("API ready after %d probe(s) (%s)", attempt, last_msg)
            return
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"API not ready after {max_wait_minutes}m on {probe}: "
                f"last probe -> {last_msg}"
            )
        log.info("api not ready (%s); sleeping %ds", last_msg,
                 probe_interval_seconds)
        time.sleep(probe_interval_seconds)


def auth_hint(exc: Exception) -> Optional[str]:
    """If an HTTP error looks like an auth issue, return a friendly hint.

    Recognizes both the raw urllib HTTPError and the richer
    OpenAIHTTPError raised by `_post_openai`.
    """
    code: Optional[int] = None
    if isinstance(exc, urllib.error.HTTPError):
        code = exc.code
    elif isinstance(exc, OpenAIHTTPError):
        code = exc.status
    if code in (401, 403):
        return ("HTTP %d — the entrance rejected the request. The script "
                "tries to flip authLevel to public automatically (see "
                "`auto_open_internal_entrance`); this likely means the "
                "flip never landed, the policy default-policy is still "
                "non-public, or the chart entrance was changed by another "
                "client mid-run. Re-run the script, or run manually:\n"
                "  olares-cli settings apps auth-level set <app> <entrance> "
                "--level public\n"
                "  olares-cli settings apps policy set <app> <entrance> "
                "--default-policy public" % code)
    return None


# --------------------------------------------------------------------------- #
# Backend-specific calls (Ollama vs OpenAI-shape vLLM)
# --------------------------------------------------------------------------- #

def pull_model_ollama(url: str, model: str, *, timeout: int) -> None:
    """Trigger a synchronous Ollama model pull. Idempotent."""
    log.info("pulling model %s (timeout=%ds)", model, timeout)
    http_post_json(f"{url}/api/pull",
                   {"name": model, "stream": False},
                   timeout=timeout)


# --- OpenAI-compatible (vLLM / llama.cpp / oai-compat) helpers ------------ #
#
# This block intentionally mirrors scripts/llm_api_benchmark.py so the two
# scripts produce comparable numbers for vLLM and LLaMA.cpp targets:
#   1. Per-call max_tokens / temperature / top_p / extra_body.
#   2. Optional Bearer auth header (chart-internal entrances usually don't
#      need it but llama-server's `--api-key` mode does).
#   3. TTFT approximation via a separate max_tokens=1 round-trip (TRUE TTFT
#      requires stream=true, which we deliberately do not do here because
#      the rest of the pipeline assumes stream=false).
#   4. Decoding of llama.cpp's `timings` block when present, so we can
#      surface the real server-side decode tokens/s instead of relying on
#      the wall-clock approximation.


@dataclass
class OpenAIConfig:
    """Per-model knobs for the openai-shape benchmark.

    All fields fall back to the global defaults from the top-level config
    so a model entry can be as short as `{"app_name": ..., "model_name": ...,
    "api_type": "openai"}` and still get sensible numbers.
    """
    api_key: str = "EMPTY"
    endpoint: str = "chat"            # "chat" -> /v1/chat/completions
                                      # "completion" -> /v1/completions
    extra_headers: dict = field(default_factory=dict)
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: Optional[float] = None
    extra_body: dict = field(default_factory=dict)
    measure_ttft_approx: bool = True


def _openai_config_from(spec: dict, cfg: dict) -> OpenAIConfig:
    """Merge per-model openai overrides on top of global config defaults."""
    g = (cfg.get("openai_defaults") or {})
    s = (spec.get("openai") or {})
    def pick(key: str, default):
        if key in s:
            return s[key]
        if key in g:
            return g[key]
        return default
    return OpenAIConfig(
        api_key=str(pick("api_key", "EMPTY")),
        endpoint=str(pick("endpoint", "chat")).lower(),
        extra_headers=dict(pick("extra_headers", {}) or {}),
        max_tokens=int(pick("max_tokens", 256)),
        temperature=float(pick("temperature", 0.0)),
        top_p=pick("top_p", None),
        extra_body=dict(pick("extra_body", {}) or {}),
        measure_ttft_approx=bool(pick("measure_ttft_approx", True)),
    )


def _openai_url(base: str, endpoint: str) -> str:
    base = base.rstrip("/")
    if endpoint == "completion":
        suffix = "/completions"
    else:
        suffix = "/chat/completions"
    if base.endswith("/v1"):
        return base + suffix
    return base + "/v1" + suffix


def _openai_headers(api_key: str, extra: dict) -> dict:
    """Build request headers. Mirrors `curl` semantics by default:
    only sends `Authorization: Bearer ...` when the user actually set
    a key — the sentinel default "EMPTY" (or empty string) is treated
    as "no auth". This avoids tripping ingress / sidecars that get
    grumpy about a Bearer header they didn't ask for.

    `extra_headers` from config can override anything here, including
    forcing an `Authorization` header back on if needed.
    """
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key and api_key.strip().upper() not in {"", "EMPTY"}:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra:
        headers.update({str(k): str(v) for k, v in extra.items()})
    return headers


def _build_openai_payload(model: str, prompt: str, conf: OpenAIConfig,
                          *, max_tokens_override: Optional[int] = None) -> dict:
    mt = (max_tokens_override
          if max_tokens_override is not None else conf.max_tokens)
    # IMPORTANT: stream=False is REQUIRED. The whole script parses the
    # response as a single JSON object; with stream=True the server replies
    # with SSE chunks (`data: {...}\n\ndata: [DONE]`) and json.loads()
    # blows up with "Expecting value: line 1 column 1 (char 0)".
    if conf.endpoint == "completion":
        payload: dict = {
            "prompt": prompt,
            "stream": False,
            "max_tokens": mt,
            "temperature": conf.temperature,
        }
    else:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": mt,
            "temperature": conf.temperature,
        }
    if conf.top_p is not None:
        payload["top_p"] = conf.top_p
    if conf.extra_body:
        payload.update(conf.extra_body)
    return payload


class OpenAIHTTPError(RuntimeError):
    """Surface a non-JSON / non-2xx response with enough context to debug.

    Message format intentionally includes status code, content-type, and
    a body snippet so an `Expecting value: ...` JSONDecodeError doesn't
    swallow whether we got an HTML login page, an empty body, or an
    actual server-side error JSON.
    """

    def __init__(self, message: str, *, status: Optional[int] = None,
                 url: str = "", content_type: str = "", body: str = ""):
        super().__init__(message)
        self.status = status
        self.url = url
        self.content_type = content_type
        self.body = body


def _decode_body(resp_or_err) -> tuple[str, str]:
    """Returns (body_text, content_type). Best-effort decode; never raises."""
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


def _post_openai(url: str, payload: dict, headers: dict,
                 *, timeout: int) -> tuple[float, dict]:
    """POST `payload` as JSON, return (wall_seconds, parsed_json_body).

    Raises OpenAIHTTPError on:
      - 4xx / 5xx (urllib HTTPError) with status, content-type and body snippet
      - 2xx with empty body
      - 2xx with non-JSON body (typical: HTML login page from a redirect,
        or an SSE stream because someone forgot stream=False)
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
        # Two extremely common gotchas worth calling out by name:
        # 1) HTML login/error page (Authelia / cluster ingress)
        # 2) SSE stream because stream=True slipped into the payload
        looks_like_html = snippet.lstrip().lower().startswith(("<!doctype", "<html"))
        looks_like_sse = snippet.startswith("data:")
        hint = ""
        if looks_like_html:
            hint = (" — looks like an HTML page; the entrance probably "
                    "isn't authenticated for this script. Try "
                    "set_public_during_run=true or set endpoint_url to "
                    "an in-cluster URL")
        elif looks_like_sse:
            hint = (" — looks like an SSE stream; payload must include "
                    "`stream: false` for this script's parser")
        raise OpenAIHTTPError(
            f"HTTP {status} from {url} returned non-JSON body "
            f"(content-type={content_type!r}): {snippet}{hint} "
            f"[json error: {exc.msg}]",
            status=status, url=url, content_type=content_type, body=raw,
        ) from None


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


def _rough_token_count(text: str) -> int:
    """Fallback token estimate when the server didn't include `usage`.

    Mirrors scripts/llm_api_benchmark.py.rough_token_count: counts CJK
    chars individually and groups latin/digit/symbol runs as one token.
    Coarse but consistent — good enough for tps comparisons across
    same-prompt runs.
    """
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    latin_only = _CJK_RE.sub(" ", text)
    latin = len(_LATIN_RE.findall(latin_only))
    return cjk + latin


def _ms_to_seconds(value: Any) -> float:
    """Convert a llama.cpp `timings` millisecond reading to seconds.

    The keys we feed in (`prompt_ms`, `predicted_ms`, `eval_ms`, ...) are
    always millisecond-typed in llama.cpp source, so we always divide by
    1000 — no heuristic. None / non-numeric becomes 0.0.
    """
    if value is None:
        return 0.0
    try:
        return float(value) / 1000.0
    except (TypeError, ValueError):
        return 0.0


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def warmup_until_ok(url: str, model: str, api_type: str,
                    openai_conf: Optional[OpenAIConfig],
                    *, request_timeout: int,
                    retries: int = 5,
                    sleep_seconds: int = 30) -> None:
    """Warm the model up, retrying on transient failures.

    `wait_until_api_ready` already verifies the server's listing
    endpoint, but vLLM in particular sometimes 503s once or twice on
    the first inference call between "models endpoint live" and "first
    real request works" (KV cache init, lazy compile, etc.). A short
    retry loop bridges that gap.

    `warmup_until_ok` keeps the original "swallow on final failure"
    semantics: if all retries are exhausted we just log a warning and
    let the real benchmark capture the error in QuestionResult, so a
    flaky model doesn't take down the whole multi-model run.
    """
    last_exc: Optional[BaseException] = None
    for i in range(1, retries + 1):
        try:
            if api_type == "openai":
                assert openai_conf is not None
                full = _openai_url(url, openai_conf.endpoint)
                # Tiny max_tokens for warmup keeps cold-start cost
                # low and predictable, regardless of conf.max_tokens.
                payload = _build_openai_payload(
                    model, "ping", openai_conf, max_tokens_override=4)
                _post_openai(
                    full, payload,
                    _openai_headers(openai_conf.api_key,
                                    openai_conf.extra_headers),
                    timeout=request_timeout,
                )
            else:
                http_post_json(
                    f"{url}/api/generate",
                    {"model": model, "prompt": "ping", "stream": False},
                    timeout=request_timeout,
                )
            log.info("warmup ok (attempt %d/%d)", i, retries)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log.warning("warmup attempt %d/%d failed: %s", i, retries, exc)
            if i < retries:
                time.sleep(sleep_seconds)
    log.warning("warmup exhausted retries (last error: %s); "
                "continuing into the benchmark anyway", last_exc)


# --------------------------------------------------------------------------- #
# Result model
# --------------------------------------------------------------------------- #

@dataclass
class QuestionResult:
    """Per-prompt timing record. Field semantics differ slightly between
    api_type=ollama and api_type=openai because the upstream APIs report
    timing differently:

      ollama (/api/generate, stream=false)
        - ttft_seconds        : load + prompt_eval (PRECISE; from server)
        - eval_seconds        : decode duration (server)
        - tps                 : decode tokens / decode seconds (server)
        - total_server_seconds: total_duration (server)

      openai-compatible (/v1/chat/completions, stream=false)
        - ttft_seconds        : APPROX. round-trip of a separate
                                max_tokens=1 request; 0 if disabled
        - eval_seconds        : llama.cpp's `timings.predicted_ms` if
                                returned by the server, else 0
        - tps                 : server tps if `timings` available,
                                else client_tps (eval_count / wall)
        - total_server_seconds: equals wall_seconds (no separate server
                                value in stream=false mode)
        - prompt_tokens / total_tokens / client_tps / server_tps_reported
          are populated from `usage` and `timings` blocks
    """
    prompt: str
    ok: bool = False
    error: Optional[str] = None
    response_chars: int = 0
    wall_seconds: float = 0.0
    ttft_seconds: float = 0.0
    load_seconds: float = 0.0
    prompt_eval_seconds: float = 0.0
    eval_count: int = 0                # generated/completion tokens
    eval_seconds: float = 0.0
    tps: float = 0.0
    total_server_seconds: float = 0.0
    # OpenAI-only extras (kept at zero for ollama rows)
    prompt_tokens: int = 0
    total_tokens: int = 0
    client_tps: float = 0.0            # completion / wall (always honest)
    server_tps_reported: float = 0.0   # only when llama.cpp returns timings
    tokens_estimated: bool = False     # True when usage was missing and we
                                       # fell back to a char-count estimate
    note: str = ""                     # explainer for partial metrics


@dataclass
class ModelResult:
    app_name: str
    model: str
    api_type: str = "ollama"
    started_at: str = ""
    finished_at: str = ""
    install_decision: str = ""        # "fresh" / "reused" / "recovered" / ""
    install_ok: bool = False
    install_seconds: float = 0.0
    uninstall_skipped: bool = False   # true when preserve_if_existed kicked in
    uninstall_ok: bool = False
    uninstall_seconds: float = 0.0
    endpoint: str = ""
    error: Optional[str] = None
    questions: list[QuestionResult] = field(default_factory=list)

    def avg(self, attr: str) -> float:
        ok_values = [getattr(q, attr) for q in self.questions if q.ok]
        return (sum(ok_values) / len(ok_values)) if ok_values else 0.0


# --------------------------------------------------------------------------- #
# Per-prompt benchmark — Ollama (/api/generate, accurate timing fields)
# --------------------------------------------------------------------------- #

def benchmark_prompt_ollama(url: str, model: str, prompt: str,
                            *, request_timeout: int) -> QuestionResult:
    payload = {"model": model, "prompt": prompt, "stream": False}
    started = time.perf_counter()
    try:
        body = http_post_json(f"{url}/api/generate", payload,
                              timeout=request_timeout)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        hint = auth_hint(exc)
        if hint:
            msg = f"{msg} ({hint})"
        return QuestionResult(
            prompt=prompt, ok=False, error=msg,
            wall_seconds=round(time.perf_counter() - started, 3),
        )
    wall = time.perf_counter() - started

    # Ollama reports durations in nanoseconds.
    load = body.get("load_duration", 0) / 1e9
    prompt_eval = body.get("prompt_eval_duration", 0) / 1e9
    eval_count = int(body.get("eval_count", 0))
    eval_dur = body.get("eval_duration", 0) / 1e9
    total = body.get("total_duration", 0) / 1e9

    return QuestionResult(
        prompt=prompt,
        ok=True,
        response_chars=len(body.get("response", "")),
        wall_seconds=round(wall, 3),
        ttft_seconds=round(load + prompt_eval, 3),
        load_seconds=round(load, 3),
        prompt_eval_seconds=round(prompt_eval, 3),
        eval_count=eval_count,
        eval_seconds=round(eval_dur, 3),
        tps=round(eval_count / eval_dur, 2) if eval_dur > 0 else 0.0,
        total_server_seconds=round(total, 3),
    )


# --------------------------------------------------------------------------- #
# Per-prompt benchmark — OpenAI shape (/v1/chat/completions, vLLM/llama.cpp)
# --------------------------------------------------------------------------- #

def _extract_openai_response(body: dict) -> dict:
    """Pull answer + token + (optional) server timings out of the response.

    Handles both Chat Completion (`choices[0].message.content`) and the
    legacy Completion shape (`choices[0].text`). llama-server's OpenAI
    endpoint additionally exposes a `timings` block with millisecond
    precision; vLLM does not, so those fields stay 0 when unavailable.
    """
    choices = body.get("choices") or []
    answer = ""
    if choices:
        first = choices[0] or {}
        msg = first.get("message")
        if isinstance(msg, dict):
            answer = msg.get("content") or ""
        else:
            answer = first.get("text") or ""

    usage = body.get("usage") or {}
    completion = usage.get("completion_tokens")
    prompt = usage.get("prompt_tokens")
    total = usage.get("total_tokens")

    timings = body.get("timings") or body.get("timing") or {}
    server_prompt = _ms_to_seconds(
        timings.get("prompt_ms") or timings.get("prompt_eval_ms"))
    server_gen = _ms_to_seconds(
        timings.get("predicted_ms") or timings.get("generation_ms")
        or timings.get("eval_ms"))
    server_tps = _to_float(
        timings.get("predicted_per_second") or timings.get("tokens_per_second"))

    return {
        "answer": answer,
        "completion_tokens": int(completion) if completion is not None else None,
        "prompt_tokens": int(prompt) if prompt is not None else None,
        "total_tokens": int(total) if total is not None else None,
        "server_prompt_eval_seconds": server_prompt,
        "server_generation_seconds": server_gen,
        "server_tps": server_tps,
    }


def _measure_openai_ttft(url: str, model: str, prompt: str,
                         conf: OpenAIConfig,
                         *, timeout: int) -> Optional[float]:
    """Approximate TTFT by issuing a max_tokens=1 request and measuring
    its round-trip. Not a true TTFT (would need stream=true) but a useful
    proxy for "how long until the first token is available". Returns None
    if the probe fails — the caller treats that as "no measurement" rather
    than a benchmark failure.
    """
    full = _openai_url(url, conf.endpoint)
    headers = _openai_headers(conf.api_key, conf.extra_headers)
    payload = _build_openai_payload(model, prompt, conf,
                                    max_tokens_override=1)
    try:
        wall, _ = _post_openai(full, payload, headers, timeout=timeout)
        return wall
    except Exception as exc:  # noqa: BLE001
        log.debug("ttft probe failed: %s", exc)
        return None


def benchmark_prompt_openai(url: str, model: str, prompt: str,
                            conf: OpenAIConfig,
                            *, request_timeout: int) -> QuestionResult:
    """OpenAI-compatible benchmark for vLLM / llama.cpp / other oai-compat
    backends. See the QuestionResult docstring for what each timing field
    means under stream=false.
    """
    full = _openai_url(url, conf.endpoint)
    headers = _openai_headers(conf.api_key, conf.extra_headers)

    ttft_approx: Optional[float] = None
    if conf.measure_ttft_approx:
        ttft_approx = _measure_openai_ttft(url, model, prompt, conf,
                                           timeout=request_timeout)

    payload = _build_openai_payload(model, prompt, conf)
    started = time.perf_counter()
    try:
        wall, body = _post_openai(full, payload, headers,
                                  timeout=request_timeout)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        hint = auth_hint(exc)
        if hint:
            msg = f"{msg} ({hint})"
        return QuestionResult(
            prompt=prompt, ok=False, error=msg,
            wall_seconds=round(time.perf_counter() - started, 3),
            ttft_seconds=round(ttft_approx, 3) if ttft_approx else 0.0,
        )

    parsed = _extract_openai_response(body)
    answer = parsed["answer"]
    completion = parsed["completion_tokens"]
    estimated = False
    if completion is None:
        completion = _rough_token_count(answer)
        estimated = True
    prompt_tokens = parsed["prompt_tokens"] or 0
    total_tokens = parsed["total_tokens"]
    if total_tokens is None and prompt_tokens and completion:
        total_tokens = prompt_tokens + completion

    server_gen = parsed["server_generation_seconds"]
    server_tps_reported = parsed["server_tps"]
    # Prefer the real server TPS when llama.cpp gives it to us, else fall
    # back to the end-to-end client TPS (this is the only honest number
    # vLLM exposes in stream=false mode).
    client_tps = (completion / wall) if wall > 0 and completion else 0.0
    if server_tps_reported and server_tps_reported > 0:
        chosen_tps = server_tps_reported
    elif server_gen and server_gen > 0 and completion:
        chosen_tps = completion / server_gen
    else:
        chosen_tps = client_tps

    notes: list[str] = []
    if ttft_approx is None and conf.measure_ttft_approx:
        notes.append("ttft probe failed; ttft_seconds=0")
    elif ttft_approx is not None:
        notes.append("ttft_seconds is APPROX (max_tokens=1 round-trip; "
                     "TRUE TTFT needs stream=true)")
    if not (server_tps_reported and server_tps_reported > 0) and \
       not (server_gen and server_gen > 0):
        notes.append("no server-side `timings` block; tps = client end-to-end "
                     "(includes prefill+decode+network)")
    if estimated:
        notes.append("usage missing; eval_count is a char-count estimate")

    return QuestionResult(
        prompt=prompt,
        ok=True,
        response_chars=len(answer),
        wall_seconds=round(wall, 3),
        ttft_seconds=round(ttft_approx, 3) if ttft_approx else 0.0,
        load_seconds=0.0,
        prompt_eval_seconds=round(parsed["server_prompt_eval_seconds"], 3),
        eval_count=int(completion or 0),
        eval_seconds=round(server_gen, 3),
        tps=round(chosen_tps, 2),
        total_server_seconds=round(wall, 3),
        prompt_tokens=prompt_tokens,
        total_tokens=int(total_tokens or 0),
        client_tps=round(client_tps, 2),
        server_tps_reported=round(server_tps_reported, 2)
            if server_tps_reported else 0.0,
        tokens_estimated=estimated,
        note="; ".join(notes),
    )


# --------------------------------------------------------------------------- #
# Per-model orchestration
# --------------------------------------------------------------------------- #

def bench_model(spec: dict, prompts: list[str], cfg: dict) -> ModelResult:
    app = spec["app_name"]
    model = spec["model_name"]
    api_type = spec.get("api_type", cfg.get("api_type", "ollama")).lower()
    res = ModelResult(app_name=app, model=model, api_type=api_type,
                      started_at=datetime.utcnow().isoformat() + "Z")

    install_minutes = int(spec.get("install_timeout_minutes",
                                   cfg.get("install_timeout_minutes", 90)))
    uninstall_minutes = int(spec.get("uninstall_timeout_minutes",
                                     cfg.get("uninstall_timeout_minutes", 30)))
    request_timeout = int(spec.get("request_timeout_seconds",
                                   cfg.get("request_timeout_seconds", 1800)))
    pull_timeout = int(spec.get("pull_timeout_seconds",
                                cfg.get("pull_timeout_seconds", 3600)))
    delete_data = bool(spec.get("delete_data", cfg.get("delete_data", True)))
    do_pull = bool(spec.get("pull_model", cfg.get("pull_model", True)))
    # Auto-flip entrance auth-level to "public" when find_entrance reports
    # it as private/internal (chart default for vLLM/llama.cpp is internal).
    # The setting is per-app and dies with the app on uninstall, so this is
    # safe by default; flip to false if you'd rather have the script error
    # out and let you change it yourself.
    auto_open = bool(spec.get("auto_open_internal_entrance",
                              cfg.get("auto_open_internal_entrance", True)))
    # Legacy alias kept so existing configs keep working: set_public_during_run
    # forces a flip even when the level is already public, which is
    # functionally the same as auto_open here.
    if "set_public_during_run" in spec or "set_public_during_run" in cfg:
        legacy = bool(spec.get("set_public_during_run",
                               cfg.get("set_public_during_run", False)))
        if legacy:
            auto_open = True
    skip_if_running = bool(spec.get("skip_install_if_running",
                                    cfg.get("skip_install_if_running", True)))
    preserve_if_existed = bool(spec.get("preserve_if_existed",
                                        cfg.get("preserve_if_existed", False)))
    # Master switch: when False, the post-benchmark uninstall is skipped
    # unconditionally (even when the model was freshly installed by this
    # run). Useful when you want the model to stay around for follow-up
    # manual testing, or when running a single-model spot check. Defaults
    # to True so the original "free GPU/disk between models" behavior
    # holds for the multi-model loop.
    uninstall_after = bool(spec.get("uninstall_after_run",
                                    cfg.get("uninstall_after_run", True)))
    install_envs = list(spec.get("envs") or [])  # per-app --env KEY=VALUE list
    # OpenAI-shape backends (vLLM / llama.cpp) get their own knob block
    # so we can pass max_tokens / temperature / api_key / extra_body etc.
    # No effect when api_type != "openai".
    openai_conf = _openai_config_from(spec, cfg)

    already_existed = False

    try:
        # 1. ensure the app is running (reuse / wait / reinstall as needed)
        t = time.perf_counter()
        already_existed, decision = ensure_installed(
            app,
            install_minutes=install_minutes,
            uninstall_minutes=uninstall_minutes,
            install_envs=install_envs,
            delete_data=delete_data,
            skip_if_running=skip_if_running,
        )
        res.install_decision = decision
        res.install_seconds = round(time.perf_counter() - t, 1)
        res.install_ok = True

        # 2. discover the API entrance (or honor an explicit override)
        entrance, url, auth_level = find_entrance(
            app,
            spec.get("entrance_name"),
            override=spec.get("endpoint_url"),
        )
        res.endpoint = url
        log.info("using entrance %s -> %s (authLevel=%s)",
                 entrance, url, auth_level or "n/a")

        # 3. make sure the entrance is reachable anonymously. For chart
        #    defaults like authLevel=internal we flip it to public so the
        #    raw HTTP request below isn't bounced to a login page. The
        #    setting is per-app and dies on uninstall.
        ensure_entrance_public(app, entrance, auth_level, auto_open=auto_open)

        # 3.5 wait for the in-container LLM server to actually answer.
        # `market install --watch` only signals chart=running; the LLM
        # server inside may still be downloading / loading weights, and
        # there's no chart-level signal for it. The wait is unbounded
        # in real terms (we just sleep + retry until the model API
        # answers OR the configured wall-clock budget expires).
        api_ready_minutes = int(spec.get(
            "api_ready_timeout_minutes",
            cfg.get("api_ready_timeout_minutes", 60),
        ))
        api_ready_interval = int(spec.get(
            "api_ready_probe_interval_seconds",
            cfg.get("api_ready_probe_interval_seconds", 30),
        ))
        wait_until_api_ready(
            url, api_type, model,
            max_wait_minutes=api_ready_minutes,
            probe_interval_seconds=api_ready_interval,
        )

        # 4. ensure the model is pulled (Ollama only; vLLM ships weights via chart)
        if api_type == "ollama" and do_pull:
            try:
                pull_model_ollama(url, model, timeout=pull_timeout)
            except Exception as exc:  # noqa: BLE001
                log.warning("pull failed (will still try generate): %s", exc)

        # 5. warmup with retry — gives the server one more grace window
        #    after the readiness probe in case of lazy-init flakiness.
        warmup_until_ok(
            url, model, api_type,
            openai_conf if api_type == "openai" else None,
            request_timeout=pull_timeout,
            retries=int(spec.get("warmup_retries",
                                 cfg.get("warmup_retries", 5))),
            sleep_seconds=int(spec.get(
                "warmup_retry_sleep_seconds",
                cfg.get("warmup_retry_sleep_seconds", 30))),
        )

        # 6. real benchmark
        for prompt in prompts:
            log.info("prompt: %s", prompt[:60].replace("\n", " "))
            if api_type == "openai":
                qr = benchmark_prompt_openai(url, model, prompt, openai_conf,
                                             request_timeout=request_timeout)
            else:
                qr = benchmark_prompt_ollama(url, model, prompt,
                                             request_timeout=request_timeout)
            res.questions.append(qr)
            if qr.ok:
                if api_type == "openai":
                    log.info("  -> wall=%.3fs ttft~%.3fs tokens=%d tps=%.2f "
                             "(client_tps=%.2f, server_tps=%.2f)",
                             qr.wall_seconds, qr.ttft_seconds, qr.eval_count,
                             qr.tps, qr.client_tps, qr.server_tps_reported)
                else:
                    log.info("  -> ttft=%.3fs tokens=%d tps=%.2f wall=%.3fs",
                             qr.ttft_seconds, qr.eval_count, qr.tps,
                             qr.wall_seconds)
            else:
                log.warning("  -> error: %s", qr.error)

    except subprocess.CalledProcessError as exc:
        res.error = (
            f"{cli()} failed (exit={exc.returncode}): "
            f"{(exc.stderr or exc.stdout or '').strip()[:500]}"
        )
        log.error(res.error)
    except Exception as exc:  # noqa: BLE001
        log.exception("model %s failed", app)
        res.error = str(exc)

    finally:
        # Default behavior: always uninstall so the next model gets the
        # GPU + disk. Two opt-outs (in order of precedence):
        #   1. uninstall_after_run=false  -> never uninstall this model.
        #   2. preserve_if_existed=true   -> keep the model only if it
        #      was already installed before this run started.
        if not uninstall_after:
            log.info("%s: uninstall_after_run=false; skipping post-benchmark "
                     "uninstall (model stays installed)", app)
            res.uninstall_skipped = True
            res.uninstall_ok = True
        elif already_existed and preserve_if_existed:
            log.info("%s was pre-existing and preserve_if_existed=true; "
                     "skipping post-benchmark uninstall", app)
            res.uninstall_skipped = True
            res.uninstall_ok = True
        else:
            try:
                t = time.perf_counter()
                market_uninstall(app, watch_minutes=uninstall_minutes,
                                 delete_data=delete_data)
                res.uninstall_seconds = round(time.perf_counter() - t, 1)
                res.uninstall_ok = True
            except Exception as exc:  # noqa: BLE001
                log.exception("uninstall %s failed", app)
                if not res.error:
                    res.error = f"uninstall: {exc}"
        res.finished_at = datetime.utcnow().isoformat() + "Z"

    return res


# --------------------------------------------------------------------------- #
# Reporting + email
# --------------------------------------------------------------------------- #

def _fmt_duration(seconds: float) -> str:
    """`123.4` -> `2m 03s`, `5.4` -> `5.4s`. For the email subtitle only."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"

def render_html(results: list[ModelResult]) -> str:
    """Email-friendly single-table summary.

    Designed to be readable in Gmail / Apple Mail / mobile clients.
    Every important style is inlined (Gmail's web client strips
    <style> blocks). Numeric columns are right-aligned with tabular
    figures so columns line up across rows. The per-prompt detail
    intentionally lives in the JSON attachment, not the email body.
    """
    total_models = len(results)
    ok_prompts = sum(sum(1 for q in r.questions if q.ok) for r in results)
    total_prompts = sum(len(r.questions) for r in results)
    failed_models = sum(
        1 for r in results
        if r.error is not None or not any(q.ok for q in r.questions)
    )
    total_wall = sum(
        q.wall_seconds for r in results for q in r.questions if q.ok
    )

    th = ("padding:8px 12px;background:#f5f6f8;"
          "border-bottom:1px solid #dcdfe4;"
          "color:#555;font-weight:600;font-size:11px;"
          "text-transform:uppercase;letter-spacing:.04em;"
          "white-space:nowrap;")
    th_l = th + "text-align:left;"
    th_c = th + "text-align:center;"
    th_r = th + "text-align:right;"

    cell = ("padding:9px 12px;border-bottom:1px solid #eef0f3;"
            "vertical-align:middle;font-size:13px;color:#222;")
    cell_l = cell + "text-align:left;"
    cell_c = cell + "text-align:center;"
    cell_r = cell + ("text-align:right;font-variant-numeric:tabular-nums;"
                     "font-feature-settings:'tnum';")

    rows: list[str] = []
    for idx, r in enumerate(results):
        ok_q = sum(1 for q in r.questions if q.ok)
        total_q = len(r.questions)
        run_ok = r.error is None and ok_q > 0
        bg = "#ffffff" if idx % 2 == 0 else "#fafbfc"

        if run_ok:
            badge = ('<span style="display:inline-block;padding:2px 9px;'
                     'border-radius:10px;background:#e6f7ee;color:#0a7d48;'
                     'font-size:11px;font-weight:600;letter-spacing:.02em">'
                     'OK</span>')
        else:
            err_full = r.error or "no successful prompt"
            badge = ('<span style="display:inline-block;padding:2px 9px;'
                     'border-radius:10px;background:#fdecea;color:#c0392b;'
                     'font-size:11px;font-weight:600;letter-spacing:.02em" '
                     f'title="{html_escape(err_full)}">FAIL</span>')

        # Install/uninstall pipeline overhead appears as a faint subtitle
        # under the app name — keeps the email "test parameters" focused
        # while still giving an at-a-glance install/cleanup signal.
        decision = r.install_decision or "-"
        install_s = r.install_seconds or 0
        if r.uninstall_skipped:
            tail = "uninstall skipped"
        elif r.uninstall_seconds:
            tail = f"uninstall {r.uninstall_seconds:.0f}s"
        else:
            tail = "uninstall n/a"
        sub = f"{html_escape(decision)} · install {install_s:.0f}s · {tail}"

        rows.append(
            f'<tr style="background:{bg};">'
            f'<td style="{cell_l}">'
            f'<div style="font-weight:600;color:#111">'
            f'{html_escape(r.app_name)}</div>'
            f'<div style="color:#888;font-size:11px;margin-top:2px">'
            f'{sub}</div>'
            f'</td>'
            f'<td style="{cell_l}">'
            f'<code style="background:#f3f4f6;padding:1px 6px;'
            f'border-radius:3px;font-size:12px;color:#1a1a1a;'
            f'font-family:SFMono-Regular,Consolas,Menlo,monospace">'
            f'{html_escape(r.model)}</code></td>'
            f'<td style="{cell_c};color:#666">{html_escape(r.api_type)}</td>'
            f'<td style="{cell_c}">{ok_q}/{total_q}</td>'
            f'<td style="{cell_r}">{r.avg("ttft_seconds"):.2f}</td>'
            f'<td style="{cell_r};font-weight:600;color:#0b5fff">'
            f'{r.avg("tps"):.1f}</td>'
            f'<td style="{cell_r}">{r.avg("eval_count"):.0f}</td>'
            f'<td style="{cell_r}">{r.avg("wall_seconds"):.2f}</td>'
            f'<td style="{cell_c}">{badge}</td>'
            "</tr>"
        )

    summary_bits = [
        f'{total_models}&nbsp;model{"s" if total_models != 1 else ""}',
        f'{ok_prompts}/{total_prompts}&nbsp;prompts&nbsp;OK',
        f'wall&nbsp;{_fmt_duration(total_wall)}',
    ]
    if failed_models:
        summary_bits.append(
            f'<span style="color:#c0392b">'
            f'{failed_models}&nbsp;failed</span>'
        )
    subtitle = " &middot; ".join(summary_bits)

    return (
        '<div style="font-family:-apple-system,BlinkMacSystemFont,'
        '\'Segoe UI\',Roboto,Helvetica,Arial,sans-serif;color:#222;'
        'max-width:920px;padding:16px 4px;line-height:1.45">'
        # header card
        '<div style="margin:0 0 14px 0">'
        '<div style="font-size:20px;font-weight:600;color:#111;'
        'margin-bottom:2px">Olares LLM benchmark</div>'
        f'<div style="color:#666;font-size:13px">'
        f'{datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}'
        f' &middot; {subtitle}</div>'
        '</div>'
        # main table — wrapped in a div so the rounded border survives
        # email clients that drop border-radius on <table>.
        '<div style="border:1px solid #e5e7eb;border-radius:8px;'
        'overflow:hidden;background:#fff">'
        '<table style="border-collapse:collapse;width:100%;'
        'font-size:13px"><thead><tr>'
        f'<th style="{th_l}">App</th>'
        f'<th style="{th_l}">Model</th>'
        f'<th style="{th_c}">API</th>'
        f'<th style="{th_c}">OK / N</th>'
        f'<th style="{th_r}">TTFT (s)</th>'
        f'<th style="{th_r}">TPS</th>'
        f'<th style="{th_r}">Tokens</th>'
        f'<th style="{th_r}">Wall (s)</th>'
        f'<th style="{th_c}">Status</th>'
        '</tr></thead><tbody>'
        + "".join(rows) +
        '</tbody></table></div>'
        # footer legend — compact, single line wherever possible
        '<p style="margin:14px 2px 0;color:#888;font-size:11.5px;'
        'line-height:1.55">'
        '<b>TTFT</b> = time to first token &middot; '
        '<b>TPS</b> = generated tokens per second &middot; '
        '<b>Tokens</b> = avg generated tokens per prompt &middot; '
        '<b>Wall</b> = client-side request &rarr; response. '
        'Numbers are averaged over successful prompts. For Ollama these '
        'come from server-reported <code style="background:#f3f4f6;'
        'padding:0 4px;border-radius:3px">load/prompt_eval/eval</code> '
        'durations; for vLLM / llama.cpp under stream=false TTFT is '
        'approximated via a max_tokens=1 round-trip and TPS prefers '
        'llama.cpp <code style="background:#f3f4f6;padding:0 4px;'
        'border-radius:3px">timings.predicted_per_second</code> when '
        'present, else completion_tokens / wall. '
        'Per-prompt records, errors, and notes are in the attached JSON.'
        '</p>'
        '</div>'
    )


def send_email(html: str, json_dump: str, cfg: dict, *, stamp: str) -> None:
    """Build the MIME message and ship it over SMTP.

    Two transport modes are supported:
      - port 465 (or `use_ssl: true`)  -> implicit TLS via `SMTP_SSL`.
      - any other port (typically 587) -> plain SMTP + STARTTLS upgrade.

    The choice matters: hitting port 465 with `smtplib.SMTP` (the old
    code path) sends a plaintext `EHLO` to a TLS-only listener, the
    server RSTs immediately, and Python surfaces it as
    `SMTPServerDisconnected: Connection unexpectedly closed` — which is
    the exact symptom we hit when `smtp_port` was bumped to 465.

    Network blips on the shared egress path are common too, so the call
    is wrapped in a small retry loop (transport errors only — auth /
    protocol errors are raised on the first try so we don't burn cycles
    on something a retry can't fix).
    """
    smtp_cfg = cfg["email"]
    msg = MIMEMultipart("mixed")
    msg["Subject"] = smtp_cfg.get(
        "subject", f"Olares LLM benchmark {datetime.utcnow().date()}")
    msg["From"] = smtp_cfg["from"]
    msg["To"] = smtp_cfg["to"]

    body = MIMEMultipart("alternative")
    body.attach(MIMEText(
        "Olares LLM benchmark results — see HTML body or attached JSON.",
        "plain", _charset="utf-8"))
    body.attach(MIMEText(html, "html", _charset="utf-8"))
    msg.attach(body)

    att = MIMEApplication(json_dump.encode("utf-8"), _subtype="json")
    att.add_header("Content-Disposition", "attachment",
                   filename=f"llm_bench_{stamp}.json")
    msg.attach(att)

    host = smtp_cfg["smtp_host"]
    port = int(smtp_cfg["smtp_port"])
    timeout = int(smtp_cfg.get("smtp_timeout", 120))
    retries = max(1, int(smtp_cfg.get("smtp_retries", 3)))
    backoff = int(smtp_cfg.get("smtp_retry_backoff", 5))
    # Explicit override beats heuristic. Heuristic: 465 -> implicit TLS,
    # everything else -> STARTTLS. This matches what every major
    # provider (Gmail, Outlook, SES, Mailgun, ...) actually advertises.
    use_ssl_cfg = smtp_cfg.get("use_ssl")
    use_ssl = bool(use_ssl_cfg) if use_ssl_cfg is not None else (port == 465)

    rcpt = [a.strip() for a in str(smtp_cfg["to"]).split(",") if a.strip()]
    raw = msg.as_string()

    last_exc: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            log.info(
                "sending email via %s:%d (ssl=%s, timeout=%ds, "
                "attempt %d/%d) to %s",
                host, port, use_ssl, timeout, attempt, retries,
                ", ".join(rcpt))
            ctx = ssl.create_default_context()
            if use_ssl:
                with smtplib.SMTP_SSL(host, port, timeout=timeout,
                                      context=ctx) as s:
                    s.ehlo()
                    s.login(smtp_cfg["username"], smtp_cfg["password"])
                    s.sendmail(smtp_cfg["from"], rcpt, raw)
            else:
                with smtplib.SMTP(host, port, timeout=timeout) as s:
                    s.ehlo()
                    s.starttls(context=ctx)
                    s.ehlo()  # re-EHLO after STARTTLS (RFC 3207)
                    s.login(smtp_cfg["username"], smtp_cfg["password"])
                    s.sendmail(smtp_cfg["from"], rcpt, raw)
            log.info("email sent")
            return
        # NOTE: order matters here. `smtplib.SMTPException` (and therefore
        # SMTPAuthenticationError, SMTPResponseException, ...) inherits
        # from OSError, so we MUST handle the protocol-level subclasses
        # first; otherwise the broad OSError catch below would swallow
        # auth failures and retry them pointlessly until the budget runs
        # out.
        except smtplib.SMTPAuthenticationError:
            # bad credentials — retrying won't help
            raise
        except (smtplib.SMTPServerDisconnected,
                smtplib.SMTPConnectError,
                TimeoutError,
                ConnectionError) as exc:
            last_exc = exc
            log.warning("smtp attempt %d/%d failed: %s: %s",
                        attempt, retries, type(exc).__name__, exc)
            if attempt < retries:
                time.sleep(min(backoff * attempt, 60))
        except smtplib.SMTPException:
            # other protocol-level error (5xx etc.) — won't be rescued
            # by a retry, surface it immediately
            raise
        except OSError as exc:
            # low-level transport error not covered by the SMTP* aliases
            # (e.g. ECONNRESET in a weird state)
            last_exc = exc
            log.warning("smtp attempt %d/%d failed: %s: %s",
                        attempt, retries, type(exc).__name__, exc)
            if attempt < retries:
                time.sleep(min(backoff * attempt, 60))

    assert last_exc is not None
    raise last_exc


# --------------------------------------------------------------------------- #
# `--probe`: list env requirements for a set of apps without installing
# --------------------------------------------------------------------------- #

def probe_apps(apps: list[str]) -> None:
    """Dump the env spec of each app so the operator can see which user
    envs (e.g. OLARES_USER_HUGGINGFACE_TOKEN) it references and which
    chart-specific envs need a `--env KEY=VAL` at install time.
    """
    for app in apps:
        try:
            info = cli_json(["market", "get", app])
        except Exception as exc:  # noqa: BLE001
            print(f"{app}: probe failed -> {exc}", file=sys.stderr)
            continue
        envs = (info or {}).get("raw_data", {}).get("envs", [])
        title = (info or {}).get("app_info", {}).get("app_entry", {}).get("title")
        version = (info or {}).get("version", "")
        print(f"\n=== {app}  ({title or '-'}, v{version}) ===")
        if not envs:
            print("  no envs declared")
            continue
        for e in envs:
            name = e.get("envName", "?")
            required = "required" if e.get("required") else "optional"
            etype = e.get("type") or "-"
            value_from = (e.get("valueFrom") or {}).get("envName")
            default = e.get("default")
            line = f"  - {name}  ({required}, type={etype})"
            if value_from:
                line += f"  <- references USER env: {value_from}"
            elif default:
                line += f"  default={default!r}"
            print(line)
            desc = e.get("description")
            if desc:
                print(f"      desc: {desc}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def setup_logging(log_path: Optional[str]) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        cfg = json.load(fp)
    if not cfg.get("models"):
        raise SystemExit("config: 'models' must be a non-empty list")
    if not cfg.get("questions"):
        raise SystemExit("config: 'questions' must be a non-empty list")
    if "email" not in cfg:
        raise SystemExit("config: 'email' section is required")
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="Olares sequential LLM benchmark")
    ap.add_argument("-c", "--config", required=True, help="path to JSON config")
    ap.add_argument("--log", help="optional log file path")
    ap.add_argument("--cli-path", help="path to the olares-cli binary "
                                       "(overrides config.cli_path; default: "
                                       "look up `olares-cli` on PATH)")
    ap.add_argument("--no-email", action="store_true",
                    help="skip the SMTP send (useful when iterating)")
    ap.add_argument("--probe", action="store_true",
                    help="only print env requirements for each app in "
                         "config.models then exit (no install / uninstall)")
    args = ap.parse_args()

    setup_logging(args.log)
    cfg = load_config(args.config)

    # CLI path resolution priority: --cli-path > config.cli_path > $PATH lookup
    set_cli_path(args.cli_path or cfg.get("cli_path") or "olares-cli")

    if args.probe:
        probe_apps([m["app_name"] for m in cfg["models"]])
        return 0

    # User-level env vars (HF token, etc.) are expected to be set ONCE,
    # out-of-band, via `olares-cli settings advanced env user set ...`.
    # The legacy `user_envs` config block is still honored here as an
    # escape hatch, but it's deliberately undocumented in the example
    # config to avoid baking secrets into the on-disk config file.
    apply_user_envs(cfg.get("user_envs") or {})

    results: list[ModelResult] = []
    for spec in cfg["models"]:
        log.info("=== model %s (%s) ===", spec["app_name"], spec["model_name"])
        results.append(bench_model(spec, cfg["questions"], cfg))
        cooldown = int(cfg.get("cooldown_seconds", 30))
        if cooldown > 0:
            log.info("cooldown %ds before next model", cooldown)
            time.sleep(cooldown)

    payload = [asdict(r) for r in results]
    out_dir = cfg.get("output_dir") or os.path.dirname(os.path.abspath(args.config))
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(out_dir, f"llm_bench_{stamp}.json")
    html_path = os.path.join(out_dir, f"llm_bench_{stamp}.html")
    json_dump = json.dumps(payload, ensure_ascii=False, indent=2)
    with open(json_path, "w", encoding="utf-8") as fp:
        fp.write(json_dump)
    html = render_html(results)
    with open(html_path, "w", encoding="utf-8") as fp:
        fp.write(html)
    log.info("wrote %s and %s", json_path, html_path)

    if args.no_email:
        log.info("--no-email set, skipping email")
        return 0

    try:
        send_email(html, json_dump, cfg, stamp=stamp)
    except Exception:  # noqa: BLE001
        log.exception("email send failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
