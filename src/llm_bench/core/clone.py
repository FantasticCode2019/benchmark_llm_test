"""Simulate the app-store *clone* flow against the v2 ``/clone`` endpoint.

Mirrors the multi-stage dance a user goes through in the market UI
(documented in ``clone.md``). The endpoint accepts a single
``POST`` URL but answers an HTTP 500 with a JSON ``backend_response``
that tells the caller what it STILL needs; the frontend reads that,
pops a dialog, and re-POSTs the same URL with the extra fields. This
module reproduces exactly that loop, driven entirely by the
caller-supplied arguments:

    POST https://market.{olaresId}.olares.com/app-store/api/v2/apps/{appName}/clone

  Round 1 ── ``{app_name, source, sync}``
             → 500, ``backend_response.code = 422``, ``type = "appEntrance"``
               the server wants a Title + a title for each entrance.
  Round 2 ── ``{.., title, entrances}``
             → 500, ``backend_response.code = 422``, ``type = "appenv"``
               the server wants the chart's required env values
               (declared in the app's ``OlaresManifest.yaml``).
  Round 3 ── ``{.., title, entrances, envs}``
             → 200, ``success = true``, an ``opID`` is returned.

The whole point of the loop is that we don't have to know up front
WHICH rounds a given app needs — a chart with no extra entrances or
no required envs simply skips the corresponding 422 and lands on 200
sooner. We start from the minimal body and only attach ``title`` /
``entrances`` / ``envs`` once the server has actually asked for them,
pulling each value from the arguments the caller passed in.

Every input the request needs — ``olares_id`` (which forms the host),
``app_name``, ``source``, ``title``, ``entrances``, ``envs``, plus the
auth material — is a function parameter. Nothing is read from global
state, so the function is trivially testable against a stub server via
the ``base_url`` override.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from llm_bench.constants import LOG_NAMESPACE
from llm_bench.exceptions import CloneError

log = logging.getLogger(LOG_NAMESPACE)


#: How many ``/clone`` POSTs we'll issue before giving up. The documented
#: flow needs at most three (entrances → envs → success); the extra
#: headroom covers a future chart that introduces another 422 stage. The
#: loop also self-terminates the moment a round can't add anything new to
#: the payload, so this is just a hard ceiling, not the usual exit path.
DEFAULT_MAX_ROUNDS = 5

#: Default per-request timeout (seconds). Clone kicks off a server-side
#: install when it finally succeeds (``sync: true``), so the final round
#: can take a while — keep this generous.
DEFAULT_TIMEOUT_SECONDS = 300


@dataclass
class CloneResult:
    """Outcome of :func:`clone_app`.

    Three terminal shapes:

      * **success** — ``success=True``; ``op_id`` / ``cloned_app_name``
        are filled from the 200 response.
      * **needs input** — ``success=False`` and ``needs_input=True``: the
        server asked for data (``requested_type`` = ``"appEntrance"`` or
        ``"appenv"``) that the caller did not supply, so the flow stopped.
        ``requested_values`` carries the server's ``missingValues`` list
        (entrance ``{name,title}`` rows, or env ``{envName,regex,...}``
        rows) so a caller can prompt for them and call again.
      * **failed** — ``success=False`` and ``needs_input=False``: the
        server returned a verdict we couldn't act on; ``error`` explains.
    """

    success: bool
    op_id: str = ""
    cloned_app_name: str = ""
    status: str = ""
    rounds: int = 0
    needs_input: bool = False
    requested_type: str = ""
    requested_values: list[dict[str, Any]] = field(default_factory=list)
    title_validation: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def _clone_url(olares_id: str, app_name: str, *,
               base_url: str | None, scheme: str) -> str:
    """Build the clone endpoint URL.

    ``base_url`` (e.g. ``http://127.0.0.1:8080``) overrides the host so
    tests can point at a stub; otherwise the canonical
    ``{scheme}://market.{olares_id}.olares.com`` host is used. The
    app-store path + raw ``app_name`` are always appended.
    """
    path = f"/app-store/api/v2/apps/{app_name}/clone"
    if base_url:
        return base_url.rstrip("/") + path
    return f"{scheme}://market.{olares_id}.olares.com" + path


def _build_headers(token: str | None, cookie: str | None,
                   extra_headers: dict[str, str] | None) -> dict[str, str]:
    """Assemble request headers, mirroring the working ``curl`` recipe::

        -H "X-Authorization: $TOKEN"
        -H "Cookie: auth_token=$TOKEN"
        -H "X-Unauth-Error: Non-Redirect"
        -H 'Content-Type: application/json'

    The app-store gateway authenticates the user from BOTH the
    ``X-Authorization`` header and the ``auth_token`` cookie, so a single
    ``token`` populates both. ``X-Unauth-Error: Non-Redirect`` is sent by
    default so an expired / missing token comes back as a JSON 401 instead
    of a 302 to the login page (an HTML redirect would blow up the JSON
    decode in :func:`_post_clone`).

    ``cookie`` lets a caller append extra cookie pairs (joined onto the
    ``auth_token`` one). ``extra_headers`` is merged last so a caller can
    override anything, including these defaults. An empty/None token or
    cookie is simply omitted.
    """
    headers = {"Content-Type": "application/json",
               "Accept": "application/json",
               "X-Unauth-Error": "Non-Redirect"}
    cookies: list[str] = []
    if token:
        headers["X-Authorization"] = token
        cookies.append(f"auth_token={token}")
    if cookie:
        cookies.append(cookie)
    if cookies:
        headers["Cookie"] = "; ".join(cookies)
    if extra_headers:
        headers.update({str(k): str(v) for k, v in extra_headers.items()})
    return headers


def _post_clone(url: str, payload: dict[str, Any],
                headers: dict[str, str], *, timeout: int,
                ) -> tuple[int, dict[str, Any]]:
    """POST ``payload`` and return ``(http_status, parsed_json_body)``.

    The documented flow returns HTTP 500 for the intermediate
    "needs more input" rounds, with the actual verdict living in the
    JSON body. ``urllib`` raises :class:`urllib.error.HTTPError` on any
    non-2xx, so we catch it and read the body off the exception — a 500
    here is expected, not fatal.

    Raises :class:`CloneError` only for failures that leave us with no
    JSON to interpret: a transport error, or a body that doesn't parse.
    """
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        # 422/500 carry the backend_response JSON we need — read it back.
        status = exc.code
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - defensive
            raw = ""
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise CloneError(
            f"clone POST {url} failed at the transport layer: {exc}"
        ) from exc

    try:
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        raise CloneError(
            f"clone POST {url} returned non-JSON body "
            f"(status={status}): {raw[:200]!r}"
        ) from exc
    if not isinstance(data, dict):
        raise CloneError(
            f"clone POST {url} returned a non-object JSON body "
            f"(status={status}): {raw[:200]!r}"
        )
    return status, data


def _is_success(body: dict[str, Any], backend_code: Any) -> bool:
    """True when the server reports the clone landed.

    Belt-and-braces: the top-level ``success`` flag, the inner
    ``backend_response.code == 200``, and ``data.status == "success"``
    all signal the same thing in the documented 200 payload; any one is
    enough.
    """
    data = body.get("data") or {}
    if body.get("success") is True:
        return True
    if backend_code == 200:
        return True
    return data.get("status") == "success" and bool(data.get("opID"))


def _extract_op(body: dict[str, Any]) -> tuple[str, str]:
    """Pull ``(op_id, cloned_app_name)`` out of a success body.

    ``opID`` shows up both at ``data.opID`` and
    ``data.backend_response.data.opID``; prefer the former and fall back.
    ``cloned_app_name`` is the server-generated unique name
    (``data.app_name``, e.g. ``windowsc9421e``).
    """
    data = body.get("data") or {}
    op_id = str(data.get("opID") or "")
    if not op_id:
        backend = (data.get("backend_response") or {}).get("data") or {}
        op_id = str(backend.get("opID") or "")
    return op_id, str(data.get("app_name") or "")


def clone_app(
    olares_id: str,
    app_name: str,
    *,
    source: str,
    title: str | None = None,
    entrances: list[dict[str, Any]] | None = None,
    envs: list[dict[str, Any]] | None = None,
    sync: bool = True,
    token: str | None = None,
    cookie: str | None = None,
    extra_headers: dict[str, str] | None = None,
    base_url: str | None = None,
    scheme: str = "https",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> CloneResult:
    """Run the staged app-store clone flow and return the outcome.

    Args:
        olares_id: The Olares ID that forms the host
            ``market.{olares_id}.olares.com`` (e.g. ``zhaoyu002`` or
            ``leamon520@olares.com`` depending on the deployment). Ignored
            when ``base_url`` is set.
        app_name: The RAW source app name to clone (e.g. ``windows``,
            ``ollamav3``). Used both in the URL and the request body.
        source: The market source the chart is pulled from, e.g.
            ``market.olares`` / ``market.test``.
        title: The new app Title the clone should carry. Sent once the
            server's first 422 (``appEntrance``) asks for it.
        entrances: New per-entrance titles, each a
            ``{"name": <entrance>, "title": <new title>}`` dict. Sent in
            the same round as ``title``.
        envs: Required chart env values, each a
            ``{"envName": <name>, "value": <value>[, "applyOnChange": bool]}``
            dict. Sent once the server's second 422 (``appenv``) asks.
        sync: Forwarded as the request body's ``sync`` flag (the UI sends
            ``true``).
        token: The logged-in user's access token. Sent as BOTH the
            ``X-Authorization`` header and the ``auth_token`` cookie (the
            gateway accepts either). Optional — omit when ``base_url``
            points at an unauthenticated stub.
        cookie: Extra cookie pair(s) appended onto the ``auth_token``
            cookie (e.g. ``"foo=bar"``).
        extra_headers: Extra headers merged last (can override defaults).
        base_url: Override the scheme+host for tests / non-standard
            deployments. When set, ``olares_id`` / ``scheme`` are unused.
        scheme: URL scheme for the canonical host (default ``https``).
        timeout: Per-request timeout in seconds.
        max_rounds: Hard ceiling on the number of POSTs.

    Returns:
        A :class:`CloneResult`. ``success=True`` on a 200; otherwise
        ``needs_input=True`` (server wanted data the caller didn't
        supply) or a plain failure with ``error`` set.

    Raises:
        CloneError: transport failure or an un-parseable / unexpected
            server response (a body that is neither success nor a
            recognised 422 stage).
    """
    url = _clone_url(olares_id, app_name, base_url=base_url, scheme=scheme)
    headers = _build_headers(token, cookie, extra_headers)

    # Which optional sections we've been ASKED to include so far. We grow
    # this set as the server reports 422s; the payload is rebuilt from it
    # every round.
    include_title = False
    include_entrances = False
    include_envs = False

    last_payload: dict[str, Any] | None = None

    for round_idx in range(1, max_rounds + 1):
        payload: dict[str, Any] = {
            "app_name": app_name,
            "source": source,
            "sync": sync,
        }
        if include_title and title is not None:
            payload["title"] = title
        if include_entrances and entrances:
            payload["entrances"] = entrances
        if include_envs and envs:
            payload["envs"] = envs

        # If a round can't add anything the previous one didn't already
        # carry, re-POSTing is pointless — the server would just ask for
        # the same thing again. This is the "caller didn't supply what the
        # server needs" exit; the needs_input result was already returned
        # below, so reaching here means a genuine stall.
        if payload == last_payload:
            return CloneResult(
                success=False,
                rounds=round_idx - 1,
                error="clone stalled: server keeps requesting data the "
                      "caller did not supply",
                raw={},
            )
        last_payload = payload

        log.info("clone %s round %d -> %s (fields: %s)",
                 app_name, round_idx, url, sorted(payload.keys()))
        status, body = _post_clone(url, payload, headers, timeout=timeout)

        data = body.get("data") or {}
        backend = data.get("backend_response") or {}
        backend_code = backend.get("code")

        if _is_success(body, backend_code):
            op_id, cloned_app_name = _extract_op(body)
            log.info("clone %s succeeded in %d round(s): opID=%s app=%s",
                     app_name, round_idx, op_id, cloned_app_name)
            return CloneResult(
                success=True,
                op_id=op_id,
                cloned_app_name=cloned_app_name,
                status=str(data.get("status") or "success"),
                rounds=round_idx,
                raw=body,
            )

        # Not success. The only recoverable case is backend_response.code
        # 422 ("needs more input"); anything else is a hard failure.
        if backend_code != 422:
            message = (body.get("message")
                       or data.get("error")
                       or f"unexpected clone response (http={status}, "
                          f"backend_code={backend_code})")
            log.warning("clone %s failed: %s", app_name, message)
            return CloneResult(
                success=False,
                status=str(data.get("status") or ""),
                rounds=round_idx,
                error=str(message),
                raw=body,
            )

        backend_data = backend.get("data") or {}
        req_type = str(backend_data.get("type") or "").strip()
        inner = backend_data.get("Data") or {}
        missing = inner.get("missingValues") or []
        title_validation = inner.get("titleValidation") or {}

        log.info("clone %s round %d: server needs %r (missing=%d)",
                 app_name, round_idx, req_type, len(missing))

        lower_type = req_type.lower()
        if "entrance" in lower_type:
            # Server wants the app Title + a title per entrance. We can
            # only proceed if the caller gave us at least one of them.
            if title is None and not entrances:
                return CloneResult(
                    success=False,
                    needs_input=True,
                    requested_type=req_type,
                    requested_values=list(missing),
                    title_validation=dict(title_validation),
                    rounds=round_idx,
                    error="server requires a Title and/or entrance titles; "
                          "none were supplied (pass title=... and/or "
                          "entrances=[...])",
                    raw=body,
                )
            include_title = True
            include_entrances = True
        elif "env" in lower_type:
            # Server wants the chart's required env values.
            if not envs:
                return CloneResult(
                    success=False,
                    needs_input=True,
                    requested_type=req_type,
                    requested_values=list(missing),
                    rounds=round_idx,
                    error="server requires env values; none were supplied "
                          "(pass envs=[{'envName':..., 'value':...}, ...])",
                    raw=body,
                )
            include_envs = True
        else:
            # A 422 stage we don't recognise — surface it so the caller
            # can inspect what the server asked for rather than spinning.
            return CloneResult(
                success=False,
                needs_input=True,
                requested_type=req_type,
                requested_values=list(missing),
                title_validation=dict(title_validation),
                rounds=round_idx,
                error=f"server requested an unrecognised input stage "
                      f"{req_type!r}",
                raw=body,
            )

    return CloneResult(
        success=False,
        rounds=max_rounds,
        error=f"clone did not complete within {max_rounds} rounds",
        raw=last_payload or {},
    )


__all__ = ["CloneResult", "clone_app"]
