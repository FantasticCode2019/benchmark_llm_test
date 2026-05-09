"""Chart lifecycle: install / uninstall / status + the state buckets
that drive ensure_installed's reuse-vs-reinstall decision.
"""
from __future__ import annotations

import json
import logging
import subprocess
from typing import Optional

from utils.cli_runner import cli, run

log = logging.getLogger("llm_bench")


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
                   envs: Optional[list] = None) -> None:
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


def get_app_state(app: str) -> Optional[dict]:
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
                     uninstall_minutes: int, install_envs: list,
                     delete_data: bool, skip_if_running: bool):
    """Make sure the app is `running` before benchmarking.

    Returns (already_existed, decision) where decision is one of
    'reused' / 'fresh' / 'recovered'.
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

    log.warning("%s in non-running state (%s); uninstall + reinstall", app, state)
    try:
        market_uninstall(app, watch_minutes=uninstall_minutes,
                         delete_data=delete_data)
    except Exception as exc:  # noqa: BLE001
        log.warning("pre-install uninstall failed (continuing): %s", exc)
    market_install(app, watch_minutes=install_minutes, envs=install_envs)
    return (False, "recovered")
