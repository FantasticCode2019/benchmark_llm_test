"""Olares sequential LLM benchmark — argparse entry point.

Invocation surfaces:
  * ``llm-bench …``              installed console script (preferred; see
                                  pyproject.toml ``[project.scripts]``)
  * ``python -m llm_bench …``    via ``src/llm_bench/__main__.py``
  * ``python3 llm_bench.py …``   legacy cron compatibility shim (root file)

All three surfaces resolve to the same :func:`main` entry below. CLI
flags are stable: -c / --log / --cli-path / --no-email / --probe.

Package layout (high level):
  llm_bench.domain    pure data: enums + result/config dataclasses
  llm_bench.core      install/uninstall, entrance, readiness, per-prompt benchmark
  llm_bench.data      config load, JSON+HTML report, SMTP send, --probe
  llm_bench.clients   ollama / vllm / openai-compat HTTP clients
  llm_bench.utils     olares-cli wrapper, HTTP helpers, formatting, token estimation
"""
from __future__ import annotations

import argparse
import logging
import os
import time

from llm_bench.constants import DEFAULT_OLARES_CLI, LOG_NAMESPACE
from llm_bench.core.orchestrator import bench_model
from llm_bench.data.config import load_config, setup_logging
from llm_bench.data.mailer import send_email
from llm_bench.data.probe import probe_apps
from llm_bench.data.report_writer import write_reports
from llm_bench.exceptions import ConfigError
from llm_bench.utils.cli_runner import set_cli_path

log = logging.getLogger(LOG_NAMESPACE)


def main() -> int:
    """Console entry: parse args, run bench loop, ship report.

    Exit codes:
      * 0 — full run succeeded (per-model errors are reported via the
        JSON / email, not via exit code; old cron behavior).
      * 1 — config error (file not found, malformed JSON, schema
        violation) OR email send failed (downstream consumers ignore
        the exit code and consult the report file, so the distinction
        is informational).
    """
    ap = argparse.ArgumentParser(
        prog="llm-bench",
        description="Olares sequential LLM benchmark",
    )
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
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        # Render without a stack trace — these are user-actionable errors,
        # the traceback would just bury the message.
        log.error("config error: %s", exc)
        return 1

    # CLI path resolution priority: --cli-path > config.cli_path > $PATH lookup.
    set_cli_path(args.cli_path or cfg.cli_path or DEFAULT_OLARES_CLI)

    if args.probe:
        probe_apps([m.app_name for m in cfg.models])
        return 0

    results = []
    for spec in cfg.models:
        log.info("=== model %s (%s) ===", spec.app_name, spec.model_name)
        results.append(bench_model(spec, cfg.questions, cfg))
        if cfg.cooldown_seconds > 0:
            log.info("cooldown %ds before next model", cfg.cooldown_seconds)
            time.sleep(cfg.cooldown_seconds)

    out_dir = cfg.output_dir or os.path.dirname(os.path.abspath(args.config))
    artifacts = write_reports(results, out_dir)
    stamp = (os.path.basename(artifacts.json_path)
             .replace("llm_bench_", "").replace(".json", ""))

    if args.no_email:
        log.info("--no-email set, skipping email")
        return 0

    try:
        send_email(cfg.email, artifacts.html, artifacts.json_dump,
                   stamp=stamp,
                   excel_bytes=artifacts.excel_bytes or None)
    except Exception:
        log.exception("email send failed")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation surface
    raise SystemExit(main())
