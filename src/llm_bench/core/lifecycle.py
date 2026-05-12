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


def market_install(app: str, *, watch_minutes: int,
                   envs: list | None = None) -> None:
    cmd = [cli(), "market", "install", app,
           "--watch", "--watch-timeout", f"{watch_minutes}m"]
    for kv in envs or []:
        cmd.extend(["--env", kv])
    run(cmd, timeout=watch_minutes * 60 + 60)


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
                     ) -> tuple[bool, InstallDecision]:
    """Make sure the app is ``running`` before benchmarking.

    Returns ``(already_existed, decision)`` where ``decision`` is a
    :class:`InstallDecision` enum member. The orchestrator stores it
    directly on ``ModelResult.install_decision``; the JSON serializer
    emits its string value so the wire format is unchanged.
    """
    row = get_app_state(app)
    if row is None:
        log.info("%s not installed, installing...", app)
        market_install(app, watch_minutes=install_minutes, envs=install_envs)
        return (False, InstallDecision.FRESH)

    state = (row.get("state") or "").strip()
    log.info("%s pre-existing state=%r op=%r", app, state, row.get("opType"))

    if state in RUNNING_STATES and skip_if_running:
        log.info("%s already running -> skipping install", app)
        return (True, InstallDecision.REUSED)

    if state in PROGRESSING_STATES:
        log.info("%s mid-lifecycle (%s); waiting for terminal state...",
                 app, state)
        market_status_watch(app, watch_minutes=install_minutes)
        row = get_app_state(app)
        new_state = (row or {}).get("state", "")
        if new_state in RUNNING_STATES:
            log.info("%s converged to running after watch", app)
            return (True, InstallDecision.REUSED)
        log.warning("%s landed in %r after watch; reinstalling", app, new_state)
        state = new_state

    if state in ROLLED_BACK_STATES:
        log.info("%s in rolled-back state %r -> reinstalling directly "
                 "(no uninstall needed: helm release already gone)",
                 app, state)
        market_install(app, watch_minutes=install_minutes, envs=install_envs)
        return (False, InstallDecision.RECOVERED)

    log.warning("%s in non-running state (%s); uninstall + reinstall", app, state)
    try:
        market_uninstall(app, watch_minutes=uninstall_minutes,
                         delete_data=delete_data)
    except Exception as exc:  # pre-install uninstall is best-effort
        log.warning("pre-install uninstall failed (continuing): %s", exc)
    market_install(app, watch_minutes=install_minutes, envs=install_envs)
    return (False, InstallDecision.RECOVERED)


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
