#!/usr/bin/env python3
"""Olares sequential LLM benchmark — entry point.

Modules:
  models.py     dataclasses (QuestionResult, ModelResult, OpenAIConfig)
  core/         install/uninstall, entrance, readiness, per-prompt benchmark
  data/         config load, JSON+HTML report, SMTP send, --probe
  utils/        olares-cli wrapper, HTTP helpers, formatting, token estimation
"""
from __future__ import annotations

import argparse
import logging
import os
import time

from core.orchestrator import bench_model
from data.config import load_config, setup_logging
from data.mailer import send_email
from data.probe import probe_apps
from data.report_writer import write_reports
from utils.cli_runner import apply_user_envs, set_cli_path

log = logging.getLogger("llm_bench")


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

    # Legacy `user_envs` block — undocumented escape hatch. The supported
    # path is to set user envs once via `olares-cli settings advanced env
    # user set ...` so secrets stay out of the config file.
    apply_user_envs(cfg.get("user_envs") or {})

    results = []
    for spec in cfg["models"]:
        log.info("=== model %s (%s) ===", spec["app_name"], spec["model_name"])
        results.append(bench_model(spec, cfg["questions"], cfg))
        cooldown = int(cfg.get("cooldown_seconds", 30))
        if cooldown > 0:
            log.info("cooldown %ds before next model", cooldown)
            time.sleep(cooldown)

    out_dir = cfg.get("output_dir") or os.path.dirname(os.path.abspath(args.config))
    json_path, _, json_dump, html = write_reports(results, out_dir)
    stamp = os.path.basename(json_path).replace("llm_bench_", "").replace(".json", "")

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
    raise SystemExit(main())
