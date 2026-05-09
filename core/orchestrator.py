"""Per-model orchestration: install -> entrance -> readiness -> warmup
-> per-prompt benchmark -> uninstall.
"""
from __future__ import annotations

import logging
import subprocess
import time
from datetime import datetime

from core.benchmark.ollama import benchmark_prompt_ollama, pull_model_ollama
from core.benchmark.openai import benchmark_prompt_openai, openai_config_from
from core.entrance import ensure_entrance_public, find_entrance
from core.lifecycle import ensure_installed, market_uninstall
from core.readiness import wait_until_api_ready, warmup_until_ok
from models import ModelResult
from utils.cli_runner import cli

log = logging.getLogger("llm_bench")


def _opt(spec: dict, cfg: dict, key: str, default):
    """Per-model `spec[key]` wins, else global `cfg[key]`, else default."""
    if key in spec:
        return spec[key]
    if key in cfg:
        return cfg[key]
    return default


def bench_model(spec: dict, prompts: list, cfg: dict) -> ModelResult:
    app = spec["app_name"]
    model = spec["model_name"]
    api_type = str(_opt(spec, cfg, "api_type", "ollama")).lower()
    res = ModelResult(app_name=app, model=model, api_type=api_type,
                      started_at=datetime.utcnow().isoformat() + "Z")

    install_minutes = int(_opt(spec, cfg, "install_timeout_minutes", 90))
    uninstall_minutes = int(_opt(spec, cfg, "uninstall_timeout_minutes", 30))
    request_timeout = int(_opt(spec, cfg, "request_timeout_seconds", 1800))
    pull_timeout = int(_opt(spec, cfg, "pull_timeout_seconds", 3600))
    delete_data = bool(_opt(spec, cfg, "delete_data", True))
    # Default False: every olares-market `ollama*` chart ships a launcher
    # that auto-pulls. Enable per-model only for bare ollama-server or
    # when you want progress logs in this script's stdout.
    do_pull = bool(_opt(spec, cfg, "pull_model", False))
    # Auto-flip entrance to public when find_entrance reports
    # private/internal. Per-app, dies on uninstall, so safe by default.
    auto_open = bool(_opt(spec, cfg, "auto_open_internal_entrance", True))
    # Legacy alias: set_public_during_run=true → force auto_open=True.
    if bool(_opt(spec, cfg, "set_public_during_run", False)):
        auto_open = True
    skip_if_running = bool(_opt(spec, cfg, "skip_install_if_running", True))
    preserve_if_existed = bool(_opt(spec, cfg, "preserve_if_existed", False))
    # Master uninstall switch (highest priority of the three skip toggles).
    uninstall_after = bool(_opt(spec, cfg, "uninstall_after_run", True))
    install_envs = list(spec.get("envs") or [])
    openai_conf = openai_config_from(spec, cfg)

    already_existed = False

    try:
        # 1. ensure the app is running
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

        # 3. flip the entrance to public if needed
        ensure_entrance_public(app, entrance, auth_level, auto_open=auto_open)

        # 3.5. wait for the model to actually be served by the backend
        # (level-1 readiness gate; same signal the chart launcher uses).
        wait_until_api_ready(
            url, api_type, model,
            max_wait_minutes=int(_opt(spec, cfg,
                                      "api_ready_timeout_minutes", 60)),
            probe_interval_seconds=int(_opt(spec, cfg,
                                            "api_ready_probe_interval_seconds",
                                            30)),
        )

        # 4. (opt-in) explicit /api/pull — only useful for bare ollama
        # daemons or to surface progress in this script's stdout. Ollama's
        # /api/pull is idempotent, so calling after step 3.5 is a no-op.
        if api_type == "ollama" and do_pull:
            try:
                pull_model_ollama(
                    url, model, timeout=pull_timeout,
                    max_attempts=int(_opt(spec, cfg,
                                          "pull_max_attempts", 5)),
                    retry_sleep_seconds=int(_opt(spec, cfg,
                                                 "pull_retry_sleep_seconds",
                                                 30)),
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("pull failed (model already present, "
                            "ignoring): %s", exc)

        # 5. warmup with retry — level-2 readiness gate (10×30s ≈ 5 min)
        warmup_until_ok(
            url, model, api_type,
            openai_conf if api_type == "openai" else None,
            request_timeout=pull_timeout,
            retries=int(_opt(spec, cfg, "warmup_retries", 10)),
            sleep_seconds=int(_opt(spec, cfg,
                                   "warmup_retry_sleep_seconds", 30)),
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
        # uninstall_after_run > preserve_if_existed > default(true)
        if not uninstall_after:
            log.info("%s: uninstall_after_run=false; skipping post-benchmark "
                     "uninstall", app)
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
