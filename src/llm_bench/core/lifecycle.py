"""Chart lifecycle: install / uninstall / status + the state buckets
that drive ensure_installed's reuse-vs-reinstall decision.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import re
import subprocess

from llm_bench.constants import LOG_NAMESPACE
from llm_bench.domain import InstallDecision
from llm_bench.utils.cli_runner import cli, run
from llm_bench.utils.time_utils import utc_now_naive

log = logging.getLogger(LOG_NAMESPACE)


# State buckets — keep in sync with cli/cmd/ctl/market/watch.go and
# framework/app-service/pkg/appstate/state_transition.go. Unknown states
# fall into the "needs reinstall" path which is the safe default.
RUNNING_STATES = {"running"}
PROGRESSING_STATES = {
    "pending", "downloading", "installing", "initializing",
    "upgrading", "applyingEnv", "resuming",
}
RECOVERABLE_STATES = {"stopped", "suspended"}

# Helm release already gone (or never landed); state machine ONLY allows
# InstallOp here — calling `uninstall` would 400. Skip it and go straight
# to install. Source: `OperationAllowedInState` in
# framework/app-service/pkg/appstate/state_transition.go.
ROLLED_BACK_STATES = {
    "installingCanceled", "pendingCanceled", "downloadingCanceled",
    "installFailed", "downloadFailed",
}


# `--env KEY=VALUE` passed to every `olares-cli market install` call.
#
# Why this lives here (and not in each caller's config): every chart
# the harness installs needs the same Hugging Face service base so
# the in-pod model-pull side-cars know where to fetch from. Forcing
# every model row to repeat it would be footgun-prone; centralising
# it here means a single edit (or env override at runtime) flips
# every install at once.
#
# Caller semantics: entries the caller passes in via ``ensure_installed
# install_envs=...`` with the SAME key OVERRIDE the default — see
# :func:`_merge_install_envs`. Set the caller-side value to an empty
# string (``"OLARES_USER_HUGGINGFACE_SERVICE="``) to forward an empty
# value to the chart instead of the default URL.
_DEFAULT_INSTALL_ENVS: tuple[str, ...] = (
    "OLARES_USER_HUGGINGFACE_SERVICE=https://huggingface.co",
)


def _merge_install_envs(envs: list | None) -> list[str]:
    """Layer caller-supplied envs on top of :data:`_DEFAULT_INSTALL_ENVS`.

    Each entry is a ``"KEY=VALUE"`` string passed straight through to
    ``olares-cli market install --env``; the CLI handles its own
    quoting so we deliberately do NOT shell-escape here. Caller
    entries with a ``KEY`` that matches a default override the
    default's value (last-write-wins semantics, defaults written
    first). Strings without an ``=`` are not deduplicated and just
    pass through verbatim — this preserves the prior behaviour where
    callers could send arbitrary ``--env`` flags through this code
    path even if they don't fit the KEY=VALUE shape (rare; typically
    a shell idiom). Non-string entries are silently dropped.

    The returned list places merged ``KEY=VALUE`` entries first
    (insertion order), then any non-conforming pass-through strings
    last; ``--env`` flag order is irrelevant to the underlying
    helm/chart machinery, so this is purely cosmetic.
    """
    keyed: dict[str, str] = {}
    passthrough: list[str] = []
    for kv in (*_DEFAULT_INSTALL_ENVS, *(envs or [])):
        if not isinstance(kv, str):
            continue
        key, eq, value = kv.partition("=")
        if eq != "=":
            passthrough.append(kv)
            continue
        keyed[key] = value
    return [f"{k}={v}" for k, v in keyed.items()] + passthrough


class MarketInstallFailed(RuntimeError):
    """Raised when ``olares-cli market install --watch`` exits non-zero.

    Carries the parsed ``-o json`` payload (when the CLI emitted one)
    so callers can distinguish a real "app name doesn't exist"
    response — ``status == "failed"`` — from a transport / timeout
    error where no JSON ever arrived. The ``ollama_multi_bench``
    harness uses this distinction to drive its per-target
    ``model_exists`` column in the email / JSON report.
    """

    def __init__(self, app: str, *, returncode: int,
                 status: dict | None, message: str) -> None:
        super().__init__(message)
        self.app = app
        self.returncode = returncode
        # Either a dict like ``{"status": "failed", "finalState": ...,
        # "finalOpType": ...}`` or None when the CLI errored before
        # printing JSON (timeout, panic, etc.).
        self.status = status


def _parse_install_status(raw: str | None) -> dict | None:
    """Decode the JSON object emitted by ``market install -o json``.

    Returns None on empty / unparseable stdout — callers treat that as
    "no machine-readable verdict, fall back to other signals". When
    the CLI emits a single-element list we unwrap it so callers don't
    have to care which shape arrived.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("could not decode market install JSON: %s",
                    raw[:200])
        return None
    if isinstance(data, list):
        data = data[0] if data else None
    return data if isinstance(data, dict) else None


def market_install(app: str, *, watch_minutes: int,
                   envs: list | None = None) -> dict | None:
    """Run ``olares-cli market install <app> --watch -o json``.

    Always passes the entries in :data:`_DEFAULT_INSTALL_ENVS` as
    ``--env`` flags (currently just
    ``OLARES_USER_HUGGINGFACE_SERVICE=https://huggingface.co``).
    Caller-supplied ``envs`` are layered on top via
    :func:`_merge_install_envs`: same-key entries override, others
    are appended.

    Returns:
        The parsed JSON status dict on success — typically
        ``{"status": "success", "finalState": "running",
        "finalOpType": "install"}``. Returns ``None`` only when the
        CLI exited 0 but emitted no JSON (older CLI builds or unusual
        installs); a return of ``None`` still means the install
        succeeded as far as the CLI is concerned.

    Raises:
        MarketInstallFailed: when the CLI exits non-zero. The
            exception carries the parsed JSON (when available) so the
            caller can tell ``status == "failed"`` (app name unknown,
            install conclusively did NOT happen) from a transport
            error where no JSON ever arrived.
        subprocess.TimeoutExpired: when the install ran longer than
            ``watch_minutes`` (the CLI was killed before producing
            any output to parse).
    """
    cmd = [cli(), "market", "install", app, "-o", "json",
           "--watch", "--watch-timeout", f"{watch_minutes}m"]
    for kv in _merge_install_envs(envs):
        cmd.extend(["--env", kv])
    proc = run(cmd, timeout=watch_minutes * 60 + 60,
               capture=True, check=False)
    status = _parse_install_status(proc.stdout)
    if proc.returncode != 0:
        # Surface a concise reason; include the parsed status so the
        # log line is useful even without the exception chain.
        reason: str
        if isinstance(status, dict):
            reason = (f"status={status.get('status')!r} "
                      f"finalState={status.get('finalState')!r} "
                      f"finalOpType={status.get('finalOpType')!r}")
        else:
            reason = (proc.stderr or proc.stdout or "")[:200].strip() \
                     or "no output"
        message = (f"market install {app} exited "
                   f"{proc.returncode}: {reason}")
        log.warning("%s", message)
        raise MarketInstallFailed(app, returncode=proc.returncode,
                                  status=status, message=message)
    return status


def market_uninstall(app: str, *, watch_minutes: int = 30,
                     delete_data: bool = True, cascade: bool = True) -> None:
    cmd = [cli(), "market", "uninstall", app,
           "--watch", "--watch-timeout", f"{watch_minutes}m"]
    if cascade:
        cmd.append("--cascade")
    if delete_data:
        cmd.append("--delete-data")
    run(cmd, timeout=watch_minutes * 60 + 60)


def get_app_state(app: str) -> dict | None:
    """Returns the current statusRow dict, or None if the app isn't installed.

    Wraps `olares-cli market status <app> -a -o json`. The CLI exits non-zero
    when the app isn't installed, so we disable check= and treat both the
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
    """Block until the app reaches a terminal state (op-agnostic)."""
    run(
        [cli(), "market", "status", app,
         "--watch", "--watch-timeout", f"{watch_minutes}m"],
        timeout=watch_minutes * 60 + 60,
    )


def ensure_installed(app: str, *, install_minutes: int,
                     uninstall_minutes: int, install_envs: list[str],
                     delete_data: bool, skip_if_running: bool,
                     ) -> tuple[bool, InstallDecision, dict | None]:
    """Make sure the app is ``running`` before benchmarking.

    Returns ``(already_existed, decision, install_status)``:

      * ``decision`` is a :class:`InstallDecision` enum member; the
        orchestrator stores it directly on
        ``ModelResult.install_decision`` and the JSON serializer
        emits its string value so the wire format is unchanged.
      * ``install_status`` is the parsed ``-o json`` payload from the
        most recent ``market install`` call — typically
        ``{"status": "success", "finalState": "running",
        "finalOpType": "install"}``. It is ``None`` on the REUSED
        path (no install command was issued) and on success paths
        where the CLI emitted no JSON. Downstream consumers that
        only need the decision can ignore it.
    """
    row = get_app_state(app)
    if row is None:
        log.info("%s not installed, installing...", app)
        status = market_install(
            app, watch_minutes=install_minutes, envs=install_envs)
        return (False, InstallDecision.FRESH, status)

    state = (row.get("state") or "").strip()
    log.info("%s pre-existing state=%r op=%r", app, state, row.get("opType"))

    if state in RUNNING_STATES and skip_if_running:
        log.info("%s already running -> skipping install", app)
        return (True, InstallDecision.REUSED, None)

    if state in PROGRESSING_STATES:
        log.info("%s mid-lifecycle (%s); waiting for terminal state...",
                 app, state)
        market_status_watch(app, watch_minutes=install_minutes)
        row = get_app_state(app)
        new_state = (row or {}).get("state", "")
        if new_state in RUNNING_STATES:
            log.info("%s converged to running after watch", app)
            return (True, InstallDecision.REUSED, None)
        log.warning("%s landed in %r after watch; reinstalling", app, new_state)
        state = new_state

    if state in ROLLED_BACK_STATES:
        log.info("%s in rolled-back state %r -> reinstalling directly "
                 "(no uninstall needed: helm release already gone)",
                 app, state)
        status = market_install(
            app, watch_minutes=install_minutes, envs=install_envs)
        return (False, InstallDecision.RECOVERED, status)

    log.warning("%s in non-running state (%s); uninstall + reinstall", app, state)
    try:
        market_uninstall(app, watch_minutes=uninstall_minutes,
                         delete_data=delete_data)
    except Exception as exc:  # pre-install uninstall is best-effort
        log.warning("pre-install uninstall failed (continuing): %s", exc)
    status = market_install(
        app, watch_minutes=install_minutes, envs=install_envs)
    return (False, InstallDecision.RECOVERED, status)


# Olares chart names are always lowercase alphanumeric (see
# market/<app>/Chart.yaml). We refuse to interpolate anything else into
# the `sh -c "ls -d /var/log/pods/*<app>*"` string used by the sudo
# discovery path — a config-derived value is still external input.
_SAFE_APP_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _sudo_run(argv: list[str], sudo_password: str, *,
              timeout: int = 30) -> subprocess.CompletedProcess:
    """`sudo -S -p '' <argv>` with the password fed via stdin.

    `-S` makes sudo read the password from stdin; `-p ''` suppresses
    the prompt so it doesn't end up on the shared stderr stream where
    a leaked log capture might pick it up. The password itself is
    passed through `input=` and never substituted into the argv.
    """
    return subprocess.run(
        ["sudo", "-S", "-p", "", *argv],
        input=sudo_password + "\n",
        capture_output=True, text=True,
        timeout=timeout, check=False,
    )


def _list_pod_dirs(app: str, sudo_password: str | None) -> list[str]:
    """Find `/var/log/pods/*<app>*` entries.

    Without sudo: `glob.glob` (works iff /var/log/pods/ is at least
    +rx for the current user; common on plain k3s, not always on
    locked-down hosts).

    With sudo: shell-out via `sh -c 'ls -d <pattern>'` — we have to
    use a shell so the wildcard expands as root, but the app name is
    pre-validated against `_SAFE_APP_RE` so nothing dangerous can land
    in that string.
    """
    if sudo_password:
        if not _SAFE_APP_RE.match(app):
            log.warning("archive_pod_logs: refusing sudo glob for unsafe "
                        "app name %r (must match %s)",
                        app, _SAFE_APP_RE.pattern)
            return []
        proc = _sudo_run(
            ["sh", "-c", f"ls -d /var/log/pods/*{app}* 2>/dev/null || true"],
            sudo_password, timeout=15,
        )
        if proc.returncode != 0:
            log.warning("archive_pod_logs %s: sudo ls failed (rc=%d, "
                        "stderr=%s)",
                        app, proc.returncode,
                        (proc.stderr or "").strip()[:300])
            return []
        return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    return sorted(glob.glob(f"/var/log/pods/*{app}*"))


def archive_pod_logs(app: str, *, output_dir: str = "/tmp",
                     timeout: int = 120,
                     sudo_password: str | None = None) -> str | None:
    """Tar+gzip every `/var/log/pods/*<app>*` directory into
    `<output_dir>/<app>_logs_<UTCstamp>.tar.gz` and return the archive
    path. Returns None when there's nothing to archive (no matching
    pods) or when tar fails.

    Modes:
      - `sudo_password=None` (default): runs glob + tar as the current
        user. Works on hosts where /var/log/pods/ is world-readable
        (or when this script is already root).
      - `sudo_password=<str>`: every privileged step (`ls`, `tar`,
        `chown` to hand the tarball back to us) goes through `sudo -S`
        with the password fed on stdin. The password never appears on
        the argv vector, in env, or in any log line.

    Implementation notes:
      - The app name is validated against `_SAFE_APP_RE` before it can
        reach a shell glob, so a config-derived value can't escape.
      - `/var/log/pods/<pod>` are usually symlinks to the live container
        log dirs; tar follows them by default → we get the actual logs.
    """
    matches = _list_pod_dirs(app, sudo_password)
    if not matches:
        log.info("archive_pod_logs %s: no pods match /var/log/pods/*%s* "
                 "(nothing to archive)", app, app)
        return None

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        log.warning("archive_pod_logs %s: cannot create %s: %s",
                    app, output_dir, exc)
        return None

    stamp = utc_now_naive().strftime("%Y%m%d_%H%M%S")
    archive = os.path.join(output_dir, f"{app}_logs_{stamp}.tar.gz")
    log.info("archiving %d pod log dir(s) for %s into %s%s",
             len(matches), app, archive,
             " (via sudo)" if sudo_password else "")
    try:
        if sudo_password:
            proc = _sudo_run(["tar", "-czf", archive, *matches],
                             sudo_password, timeout=timeout)
        else:
            proc = subprocess.run(
                ["tar", "-czf", archive, *matches],
                capture_output=True, text=True,
                timeout=timeout, check=False,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("archive_pod_logs %s: tar invocation failed: %s",
                    app, exc)
        return None
    if proc.returncode != 0:
        log.warning("archive_pod_logs %s: tar exit %d (stderr: %s)",
                    app, proc.returncode,
                    (proc.stderr or "").strip()[:300])
        # Best-effort cleanup so a later sender doesn't email a
        # half-written tarball. If sudo wrote it, only sudo can rm it.
        try:
            if sudo_password:
                _sudo_run(["rm", "-f", archive], sudo_password, timeout=10)
            else:
                os.remove(archive)
        except (OSError, subprocess.TimeoutExpired):
            pass
        return None

    # tar-as-root produces a root-owned tarball that the benchmark
    # process can't email or scp without yet another sudo. Hand it back.
    if sudo_password:
        try:
            chown = _sudo_run(
                ["chown", f"{os.getuid()}:{os.getgid()}", archive],
                sudo_password, timeout=10,
            )
            if chown.returncode != 0:
                log.warning("archive_pod_logs %s: chown back failed "
                            "(rc=%d): %s. Archive at %s is still root-owned.",
                            app, chown.returncode,
                            (chown.stderr or "").strip()[:200], archive)
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.warning("archive_pod_logs %s: chown back raised: %s",
                        app, exc)
    return archive
