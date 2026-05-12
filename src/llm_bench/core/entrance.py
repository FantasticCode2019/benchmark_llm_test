"""Entrance discovery + auth-level management.

Picks the right entrance for an app, falls back to a cluster-internal URL
when there's no public domain, and flips authLevel + default-policy to
"public" so anonymous HTTP works.
"""
from __future__ import annotations

import logging
import time

from llm_bench.constants import LOG_NAMESPACE
from llm_bench.utils.cli_runner import cli, cli_json, run

log = logging.getLogger(LOG_NAMESPACE)


def _normalize_url(raw: str) -> str:
    """Add https:// if the entrance URL came back bare-host."""
    raw = (raw or "").strip().rstrip("/")
    if not raw:
        return ""
    if raw.startswith(("http://", "https://")):
        return raw
    return "https://" + raw


def _cluster_url_from_ports(info: dict) -> str | None:
    """Build an in-cluster URL from `apps get`'s ports[] + namespace.

    Used when the entrance has authLevel=internal (no public domain).
    Picks the first TCP port; returns None if there's nothing usable.
    """
    namespace = (info or {}).get("namespace", "")
    for p in (info or {}).get("ports") or []:
        host = (p or {}).get("host", "")
        port = (p or {}).get("port", 0)
        proto = (p or {}).get("protocol", "tcp") or "tcp"
        if not host or not port or proto.lower() != "tcp":
            continue
        if namespace:
            return f"http://{host}.{namespace}:{port}"
        return f"http://{host}:{port}"
    return None


def find_entrance(app: str, hint: str | None,
                  *, override: str | None = None):
    """Pick the entrance to hit. Returns (entrance_name, base_url, auth_level).

    auth_level is "public" / "private" / "internal" / "" (override case).

    Resolution order (each step short-circuits as soon as it succeeds):
    1. Explicit per-model `endpoint_url` override — caller wins.
    2. `settings apps get <app>` -> entrances[i].url
       (only /api/myapps runs GenEntranceURL; the dedicated /entrances
       endpoint returns app.Spec.Entrances raw with url="", so we MUST
       NOT rely on `apps entrances list` for the URL.)
    3. Cluster-internal URL from ports[] + namespace (typical when the
       chart only ships an authLevel=internal entrance with a service
       port). Treated as "internal" so callers know it bypasses Authelia.
    """
    if override:
        log.info("%s: using endpoint_url override %s", app, override)
        return (hint or "override", override.rstrip("/"), "")

    info = cli_json(["settings", "apps", "get", app])
    rows = (info or {}).get("entrances") or []
    if not rows:
        # Last-resort fallback: surface the entrance NAME at least.
        rows = cli_json(["settings", "apps", "entrances", "list", app]) or []
        if not isinstance(rows, list):
            rows = []
    if not rows:
        raise RuntimeError(f"no entrances reported for app {app!r}")

    def _pick(entries: list) -> dict:
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

    if info:
        internal = _cluster_url_from_ports(info)
        if internal:
            chosen = _pick(rows)
            log.warning("%s: no public entrance URL; falling back to "
                        "in-cluster service URL %s", app, internal)
            return (chosen.get("name") or "entrance", internal, "internal")

    raise RuntimeError(
        f"no reachable entrance for app {app!r}: entrances={rows}, "
        f"namespace={(info or {}).get('namespace')!r}, "
        f"ports={(info or {}).get('ports')}. "
        "Set `endpoint_url` for this model in the config to point at the "
        "API directly (e.g. http://<svc>.<ns>:<port>), or enable "
        "`auto_open_internal_entrance: true`."
    )


def get_entrance_auth_level(app: str, entrance: str) -> str | None:
    """Re-read the live authLevel of one entrance via `apps get -o json`.
    Returns None if the entrance is gone (or apps get failed).
    """
    try:
        info = cli_json(["settings", "apps", "get", app])
    except Exception as exc:
        log.debug("apps get %s failed: %s", app, exc)
        return None
    for r in (info or {}).get("entrances") or []:
        if r.get("name") == entrance:
            return (r.get("authLevel") or "").lower() or None
    return None


def open_entrance(app: str, entrance: str, *, verify_timeout: int = 30,
                  poll_interval: float = 2.0) -> None:
    """Flip an entrance's auth-level + default-policy to "public", then
    POLL `apps get` for up to verify_timeout seconds — the controller
    takes a beat to apply the change to the ingress.
    """
    log.info("%s/%s: flipping auth-level + policy to public", app, entrance)
    run([cli(), "settings", "apps", "auth-level", "set",
         app, entrance, "--level", "public"], timeout=60)
    run([cli(), "settings", "apps", "policy", "set",
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
        f"after {verify_timeout}s. Re-run, or flip manually:\n"
        f"  olares-cli settings apps auth-level set {app} {entrance} --level public\n"
        f"  olares-cli settings apps policy set {app} {entrance} --default-policy public"
    )


def ensure_entrance_public(app: str, entrance: str, current_level: str,
                           *, auto_open: bool) -> str:
    """Make sure the entrance is `public` so unauthenticated HTTP works."""
    level = (current_level or "").lower()
    if level == "public" or level == "":
        return level
    if not auto_open:
        raise RuntimeError(
            f"entrance {app}/{entrance} has authLevel={level!r} but "
            "auto_open_internal_entrance=false. Either flip the entrance "
            "to public manually, or set `endpoint_url` in the model "
            "config to point at an in-cluster URL that bypasses the "
            "ingress.\n"
            f"  olares-cli settings apps auth-level set {app} {entrance} --level public\n"
            f"  olares-cli settings apps policy set {app} {entrance} --default-policy public"
        )
    open_entrance(app, entrance)
    return "public"
