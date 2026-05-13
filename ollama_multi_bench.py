#!/usr/bin/env python3
"""Multi-model ollama install + sequential multi-round prompt + TTFT report.

Workflow:
    1. Read a JSON config that declares N (>=1) ollama targets plus the
       prompts and lifecycle knobs.
    2. Install every target **sequentially** — one
       `olares-cli market install --watch` after another. Already-running
       apps are reused (override with ``skip_install_if_running: false``).
    3. After all installs converge, sequentially per target:
         a. Resolve the entrance + flip to ``public`` if needed.
         b. Poll ``/api/tags`` until the configured model is loaded.
         c. Probe ``/api/show`` once to record whether the model exposes
            the ``thinking`` capability (drives per-prompt ``think:true``
            streaming TTFT probes inside ``benchmark_prompt_ollama``).
         d. Run every prompt in ``config.prompts`` in order; each round
            logs TTFT / wall / tokens / tps.
    4. Print a TTFT-focused summary table (stdout).
    5. Optionally market-uninstall every target afterwards
       (``uninstall_after: true``).

Run:
    python3 ollama_multi_bench.py -c ollama_multi_bench.example.json

The script is a focused TTFT harness — it intentionally does NOT call
``bench_model`` / ``write_reports`` / ``send_email``. It's meant to be
run interactively, not from cron. See ``llm_bench.py`` for the full
report-and-email pipeline.

Each ``/api/generate`` call is independent (stateless): the "N rounds"
are N separate POSTs. If you need a real multi-turn chat (shared
``messages`` history) wire ``/api/chat`` in instead — the ollama
benchmark client doesn't ship that path today.
"""
from __future__ import annotations

import argparse
import dataclasses
import html as html_lib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

# Make the in-tree package importable without `pip install -e .`.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from llm_bench.clients.ollama_client import ollama_supports_thinking  # noqa: E402
from llm_bench.constants import DEFAULT_OLARES_CLI, LOG_NAMESPACE  # noqa: E402
from llm_bench.core.benchmark.ollama import benchmark_prompt_ollama  # noqa: E402
from llm_bench.core.entrance import ensure_entrance_public, find_entrance  # noqa: E402
from llm_bench.core.lifecycle import ensure_installed, market_uninstall  # noqa: E402
from llm_bench.core.readiness import wait_until_api_ready  # noqa: E402
from llm_bench.data.config import setup_logging  # noqa: E402
from llm_bench.data.mailer import send_email  # noqa: E402
from llm_bench.domain import EmailConfig, QuestionResult  # noqa: E402
from llm_bench.exceptions import ConfigValidationError  # noqa: E402
from llm_bench.utils.cli_runner import set_cli_path  # noqa: E402
from llm_bench.utils.time_utils import utc_now_naive  # noqa: E402

log = logging.getLogger(LOG_NAMESPACE)


DEFAULT_PROMPTS: list[str] = [
    "你好，请用一句话介绍下你自己。",
    "1+1 等于几？请说明你的推理过程。",
    "用一句话总结你上一回合的回答。",
]


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


@dataclass
class TargetModel:
    """One install + benchmark target (app name + ollama model id)."""
    app_name: str
    model_name: str


@dataclass
class BenchConfig:
    """Everything the harness needs, decoded from the JSON config.

    All knobs match the names used by ``llm_bench`` config so a user
    familiar with the main tool reads the same vocabulary here.
    """
    targets: list[TargetModel]
    prompts: list[str]
    install_minutes: int = 90
    uninstall_minutes: int = 30
    request_timeout_seconds: int = 1800
    readiness_probe_interval_seconds: float = 2.0
    skip_install_if_running: bool = True
    delete_data: bool = True
    uninstall_after: bool = False
    cli_path: str | None = None
    log_file: str | None = None
    # Optional SMTP block. When None the script just prints the summary
    # to stdout; when populated, the same summary is also mailed.
    email: EmailConfig | None = None

    @classmethod
    def from_dict(cls, raw: Any) -> BenchConfig:
        if not isinstance(raw, dict):
            raise SystemExit(
                f"config root must be a JSON object, "
                f"got {type(raw).__name__}"
            )

        models_raw = raw.get("models")
        if not isinstance(models_raw, list) or not models_raw:
            raise SystemExit(
                "config.models[] is required and must be a non-empty list"
            )
        targets: list[TargetModel] = []
        for i, m in enumerate(models_raw):
            if not isinstance(m, dict):
                raise SystemExit(
                    f"config.models[{i}] must be an object, "
                    f"got {type(m).__name__}"
                )
            app = m.get("app_name")
            model = m.get("model_name")
            if not isinstance(app, str) or not app:
                raise SystemExit(
                    f"config.models[{i}].app_name is required (string)"
                )
            if not isinstance(model, str) or not model:
                raise SystemExit(
                    f"config.models[{i}].model_name is required (string)"
                )
            targets.append(TargetModel(app_name=app, model_name=model))

        prompts_raw = raw.get("prompts")
        if prompts_raw is None:
            prompts = list(DEFAULT_PROMPTS)
        else:
            if not isinstance(prompts_raw, list) or not prompts_raw:
                raise SystemExit(
                    "config.prompts must be a non-empty list of strings "
                    "(omit the key entirely to use the built-in defaults)"
                )
            prompts = [str(p) for p in prompts_raw]

        email_raw = raw.get("email")
        email_cfg: EmailConfig | None = None
        if email_raw is not None:
            if not isinstance(email_raw, dict):
                raise SystemExit(
                    f"config.email must be a JSON object, "
                    f"got {type(email_raw).__name__}"
                )
            try:
                email_cfg = EmailConfig.from_dict(email_raw)
            except ConfigValidationError as exc:
                raise SystemExit(f"config.email: {exc}") from exc

        return cls(
            targets=targets,
            prompts=prompts,
            install_minutes=_int(raw, "install_minutes", 90),
            uninstall_minutes=_int(raw, "uninstall_minutes", 30),
            request_timeout_seconds=_int(
                raw, "request_timeout_seconds", 1800),
            readiness_probe_interval_seconds=_float(
                raw, "readiness_probe_interval_seconds", 2.0),
            skip_install_if_running=_bool(
                raw, "skip_install_if_running", True),
            delete_data=_bool(raw, "delete_data", True),
            uninstall_after=_bool(raw, "uninstall_after", False),
            cli_path=_optional_str(raw, "cli_path"),
            log_file=_optional_str(raw, "log_file"),
            email=email_cfg,
        )


def _int(raw: dict, key: str, default: int) -> int:
    v = raw.get(key)
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"config.{key} must be an integer, got {v!r}: {exc}"
        ) from exc


def _float(raw: dict, key: str, default: float) -> float:
    v = raw.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"config.{key} must be a number, got {v!r}: {exc}"
        ) from exc


def _bool(raw: dict, key: str, default: bool) -> bool:
    v = raw.get(key)
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    raise SystemExit(
        f"config.{key} must be a boolean, got {v!r}"
    )


def _optional_str(raw: dict, key: str) -> str | None:
    v = raw.get(key)
    if v is None:
        return None
    if not isinstance(v, str):
        raise SystemExit(
            f"config.{key} must be a string, got {type(v).__name__}"
        )
    return v or None


def load_config(path: str) -> BenchConfig:
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError as exc:
        raise SystemExit(f"config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"config file is not valid JSON: {path} ({exc})"
        ) from exc
    return BenchConfig.from_dict(raw)


# ---------------------------------------------------------------------------
# Outcome records
# ---------------------------------------------------------------------------


@dataclass
class InstallOutcome:
    """Result of one ``ensure_installed`` call. Folded into the
    corresponding :class:`BenchOutcome` before reporting.
    """
    target: TargetModel
    duration_seconds: float
    decision: str
    already_existed: bool
    error: str | None = None


@dataclass
class BenchOutcome:
    """Aggregated per-target record — install + readiness + every prompt
    round. ``install_error`` non-None means we never got to readiness;
    ``error`` non-None means we got past install but failed readiness /
    entrance setup. Per-prompt failures live on each ``QuestionResult``.
    """
    target: TargetModel
    install_decision: str = ""
    install_duration_seconds: float = 0.0
    install_already_existed: bool = False
    install_error: str | None = None
    entrance_url: str = ""
    supports_thinking: bool | None = None
    rounds: list[QuestionResult] = field(default_factory=list)
    error: str | None = None

    @property
    def install_ok(self) -> bool:
        return self.install_error is None

    @property
    def reached_prompts(self) -> bool:
        return self.install_ok and self.error is None

    @property
    def any_prompt_failed(self) -> bool:
        return any(not q.ok for q in self.rounds)

    @property
    def fully_ok(self) -> bool:
        return (self.reached_prompts
                and bool(self.rounds)
                and not self.any_prompt_failed)


# ---------------------------------------------------------------------------
# Step A — sequential install
# ---------------------------------------------------------------------------


def _install_one(target: TargetModel, cfg: BenchConfig) -> InstallOutcome:
    started = time.perf_counter()
    try:
        already_existed, decision = ensure_installed(
            target.app_name,
            install_minutes=cfg.install_minutes,
            uninstall_minutes=cfg.uninstall_minutes,
            install_envs=[],
            delete_data=cfg.delete_data,
            skip_if_running=cfg.skip_install_if_running,
        )
        return InstallOutcome(
            target=target,
            duration_seconds=round(time.perf_counter() - started, 1),
            decision=decision.value,
            already_existed=already_existed,
        )
    except Exception as exc:
        log.exception("install %s failed", target.app_name)
        return InstallOutcome(
            target=target,
            duration_seconds=round(time.perf_counter() - started, 1),
            decision="failed",
            already_existed=False,
            error=str(exc),
        )


def install_sequential(cfg: BenchConfig) -> list[InstallOutcome]:
    """Install each target one after the other (no parallelism).

    Sequential is the right default here: ``olares-cli market install
    --watch`` already streams its own progress, so interleaving two
    installs only muddies the log. The next install starts as soon as
    the previous one converges to a terminal state.
    """
    log.info("=== installing %d apps sequentially ===", len(cfg.targets))
    outcomes: list[InstallOutcome] = []
    for t in cfg.targets:
        log.info("[install] %s starting...", t.app_name)
        outcome = _install_one(t, cfg)
        outcomes.append(outcome)
        if outcome.error:
            log.error("[install] %s FAILED in %.1fs: %s",
                      t.app_name, outcome.duration_seconds, outcome.error)
        else:
            log.info("[install] %s done in %.1fs "
                     "(decision=%s, already_existed=%s)",
                     t.app_name, outcome.duration_seconds,
                     outcome.decision, outcome.already_existed)
    return outcomes


# ---------------------------------------------------------------------------
# Step B — per-target: entrance → readiness → prompts
# ---------------------------------------------------------------------------


def bench_one_app(outcome: BenchOutcome,
                  cfg: BenchConfig) -> BenchOutcome:
    """Resolve the entrance, wait for the model, then run prompts in order.

    Expects ``outcome`` to already carry the install-phase fields
    (``install_decision`` / ``install_duration_seconds`` / ...). Errors
    during entrance / readiness are captured on ``outcome.error`` and
    short-circuit *this* target's run; the caller still benches the
    remaining targets.
    """
    target = outcome.target
    log.info("=== benchmarking %s (%s) ===",
             target.app_name, target.model_name)
    try:
        entrance, url, auth_level = find_entrance(
            target.app_name, hint=None, override=None)
        ensure_entrance_public(target.app_name, entrance,
                               auth_level or "", auto_open=True)
        wait_until_api_ready(
            url, "ollama", target.model_name,
            probe_interval_seconds=cfg.readiness_probe_interval_seconds,
        )
        outcome.entrance_url = url
    except Exception as exc:
        log.exception("readiness failed for %s", target.app_name)
        outcome.error = f"readiness: {exc}"
        return outcome

    try:
        outcome.supports_thinking = ollama_supports_thinking(
            outcome.entrance_url, timeout=cfg.request_timeout_seconds)
        log.info("[probe] %s supports_thinking=%s",
                 target.app_name, outcome.supports_thinking)
    except Exception as exc:
        log.warning("[probe] %s supports_thinking failed: %s",
                    target.app_name, exc)

    thinking = bool(outcome.supports_thinking)
    for i, prompt in enumerate(cfg.prompts, start=1):
        snippet = prompt[:60].replace("\n", " ")
        log.info("[round %d/%d] %s <- %s",
                 i, len(cfg.prompts), target.app_name, snippet)
        qr = benchmark_prompt_ollama(
            outcome.entrance_url, target.model_name, prompt,
            request_timeout=cfg.request_timeout_seconds,
            thinking=thinking,
        )
        outcome.rounds.append(qr)
        if qr.ok:
            think = ""
            if (qr.thinking_ttft_seconds
                    and qr.thinking_ttft_seconds != qr.ttft_seconds):
                think = f" think_ttft={qr.thinking_ttft_seconds:.3f}s"
            log.info("  -> ttft=%.3fs%s wall=%.3fs tokens=%d tps=%.2f",
                     qr.ttft_seconds, think, qr.wall_seconds,
                     qr.eval_count, qr.tps)
        else:
            log.warning("  -> error: %s", qr.error)
    return outcome


# ---------------------------------------------------------------------------
# Step C — stdout TTFT summary
# ---------------------------------------------------------------------------


def print_summary(outcomes: list[BenchOutcome]) -> None:
    """Stdout-friendly TTFT table. Stays out of the logger on purpose so
    ``--log <file>`` only captures the structured progress trace.
    """
    print()
    print("=" * 78)
    print("TTFT summary")
    print("=" * 78)
    for o in outcomes:
        header = f"{o.target.app_name} ({o.target.model_name})"

        if o.install_error:
            print(f"\n[{header}] INSTALL FAILED in "
                  f"{o.install_duration_seconds:.1f}s: {o.install_error}")
            continue

        print(f"\n[{header}]")
        print(f"  install: {o.install_decision} in "
              f"{o.install_duration_seconds:.1f}s "
              f"(already_existed={o.install_already_existed})")

        if o.error:
            print(f"  READINESS FAILED: {o.error}")
            continue

        oks = [q for q in o.rounds if q.ok]
        ttfts = [q.ttft_seconds for q in oks if q.ttft_seconds]
        thinking_ttfts = [q.thinking_ttft_seconds for q in oks
                          if q.thinking_ttft_seconds]
        print(f"  supports_thinking = {o.supports_thinking}")
        print(f"  rounds={len(o.rounds)}  ok={len(oks)}  "
              f"err={len(o.rounds) - len(oks)}")
        if ttfts:
            avg = sum(ttfts) / len(ttfts)
            print(f"  ttft           min={min(ttfts):.3f}s  "
                  f"avg={avg:.3f}s  max={max(ttfts):.3f}s")
        if thinking_ttfts:
            avg = sum(thinking_ttfts) / len(thinking_ttfts)
            print(f"  thinking_ttft  min={min(thinking_ttfts):.3f}s  "
                  f"avg={avg:.3f}s  max={max(thinking_ttfts):.3f}s")
        print(f"  {'#':>3}  {'ok':>3}  {'ttft':>9}  {'think':>9}  "
              f"{'wall':>9}  {'tokens':>6}  {'tps':>7}")
        for i, q in enumerate(o.rounds, start=1):
            if q.ok:
                print(f"  {i:>3}  {'y':>3}  "
                      f"{q.ttft_seconds:>8.3f}s  "
                      f"{q.thinking_ttft_seconds:>8.3f}s  "
                      f"{q.wall_seconds:>8.3f}s  "
                      f"{q.eval_count:>6d}  "
                      f"{q.tps:>7.2f}")
            else:
                err = (q.error or "")[:60]
                print(f"  {i:>3}  {'N':>3}  err: {err}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Step C2 — JSON dump + HTML report (also feeds the email body)
# ---------------------------------------------------------------------------


def render_json_dump(outcomes: list[BenchOutcome], cfg: BenchConfig,
                     *, stamp: str) -> str:
    """Serialize every outcome (install + readiness + all rounds + errors)
    into a JSON document. Used both as the email attachment and as the
    record-keeping artifact when ``output_dir`` is set in a future
    iteration.
    """
    return json.dumps({
        "stamp": stamp,
        "prompts": list(cfg.prompts),
        "outcomes": [dataclasses.asdict(o) for o in outcomes],
    }, ensure_ascii=False, indent=2, default=str)


_HTML_STYLE = """\
body {font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      margin: 24px; color:#222;}
h1 {margin-bottom:4px;}
h2 {margin-top:32px; border-bottom:1px solid #ddd; padding-bottom:4px;}
.meta {color:#666; font-size:0.9em;}
.tag {display:inline-block; padding:2px 8px; border-radius:10px;
      font-size:0.85em; margin-right:6px;}
.tag-ok {background:#e6f4ea; color:#137333;}
.tag-warn {background:#fef7e0; color:#a05a00;}
.tag-err {background:#fce8e6; color:#a50e0e;}
.ttft {color:#1a73e8; font-variant-numeric: tabular-nums;}
table {border-collapse:collapse; margin-top:8px; min-width:520px;
       font-variant-numeric: tabular-nums;}
th,td {border:1px solid #ddd; padding:4px 8px; text-align:right;
       font-size:0.9em;}
th {background:#f6f6f6;}
td.prompt {text-align:left; max-width:280px; word-break:break-word;}
td.err {text-align:left; background:#fdecea; color:#a50e0e;}
.error-banner {background:#fdecea; color:#a50e0e; border:1px solid #fbb;
               padding:8px 12px; border-radius:4px; margin:6px 0;}
"""


def _h(s: object) -> str:
    """Shorthand for HTML-escape."""
    return html_lib.escape(str(s), quote=True)


def _agg(values: list[float]) -> str:
    if not values:
        return "—"
    return (f"min={min(values):.3f}s · "
            f"avg={sum(values) / len(values):.3f}s · "
            f"max={max(values):.3f}s")


def render_email_html(outcomes: list[BenchOutcome], cfg: BenchConfig,
                      *, stamp: str) -> str:
    """Build a self-contained HTML report for the email body."""
    total = len(outcomes)
    fully_ok = sum(1 for o in outcomes if o.fully_ok)
    install_failed = sum(1 for o in outcomes if o.install_error)
    readiness_failed = sum(1 for o in outcomes
                           if o.install_ok and o.error)
    prompt_failed = sum(1 for o in outcomes
                        if o.reached_prompts and o.any_prompt_failed)

    parts: list[str] = []
    parts.append("<!doctype html><html><head>"
                 "<meta charset='utf-8'>"
                 "<title>Ollama multi-bench</title>"
                 f"<style>{_HTML_STYLE}</style>"
                 "</head><body>")
    parts.append("<h1>Ollama multi-bench</h1>")
    parts.append(f"<p class='meta'>stamp <code>{_h(stamp)}</code> · "
                 f"{total} target(s) · {fully_ok} fully OK · "
                 f"{install_failed} install fail · "
                 f"{readiness_failed} readiness fail · "
                 f"{prompt_failed} with prompt errors</p>")
    parts.append("<p class='meta'>prompts: <ol>")
    for p in cfg.prompts:
        parts.append(f"<li>{_h(p)}</li>")
    parts.append("</ol></p>")

    for o in outcomes:
        t = o.target
        parts.append(f"<h2>{_h(t.app_name)} "
                     f"<span class='meta'>({_h(t.model_name)})</span></h2>")

        # Status tag row.
        tags: list[str] = []
        if o.install_error:
            tags.append("<span class='tag tag-err'>install failed</span>")
        else:
            tags.append(f"<span class='tag tag-ok'>install "
                        f"{_h(o.install_decision)} "
                        f"in {o.install_duration_seconds:.1f}s</span>")
        if not o.install_error:
            if o.error:
                tags.append("<span class='tag tag-err'>readiness failed"
                            "</span>")
            else:
                think_label = ("yes" if o.supports_thinking
                               else "no" if o.supports_thinking is False
                               else "unknown")
                tags.append(f"<span class='tag tag-ok'>ready · thinking="
                            f"{_h(think_label)}</span>")
                if o.any_prompt_failed:
                    tags.append("<span class='tag tag-warn'>"
                                f"{sum(1 for q in o.rounds if not q.ok)}"
                                " prompt error(s)</span>")
        parts.append("<p>" + " ".join(tags) + "</p>")

        # Failure banners (install or readiness) short-circuit prompt table.
        if o.install_error:
            parts.append("<div class='error-banner'><strong>install error: "
                         f"</strong>{_h(o.install_error)}</div>")
            continue
        if o.error:
            parts.append("<div class='error-banner'><strong>readiness "
                         f"error: </strong>{_h(o.error)}</div>")
            continue
        if o.entrance_url:
            parts.append(f"<p class='meta'>entrance: "
                         f"<code>{_h(o.entrance_url)}</code></p>")

        # Aggregates.
        oks = [q for q in o.rounds if q.ok]
        ttfts = [q.ttft_seconds for q in oks if q.ttft_seconds]
        think_ttfts = [q.thinking_ttft_seconds for q in oks
                       if q.thinking_ttft_seconds]
        parts.append("<p class='meta'>"
                     f"ttft <span class='ttft'>{_h(_agg(ttfts))}</span>"
                     f" &nbsp;|&nbsp; thinking_ttft "
                     f"<span class='ttft'>{_h(_agg(think_ttfts))}</span>"
                     "</p>")

        # Per-round table.
        parts.append("<table><thead><tr>"
                     "<th>#</th><th>prompt</th><th>ok</th>"
                     "<th>ttft</th><th>thinking_ttft</th>"
                     "<th>wall</th><th>tokens</th><th>tps</th>"
                     "</tr></thead><tbody>")
        for i, q in enumerate(o.rounds, start=1):
            prompt_cell = _h((q.prompt or "")[:140])
            if q.ok:
                parts.append(
                    f"<tr><td>{i}</td>"
                    f"<td class='prompt'>{prompt_cell}</td>"
                    f"<td>ok</td>"
                    f"<td>{q.ttft_seconds:.3f}s</td>"
                    f"<td>{q.thinking_ttft_seconds:.3f}s</td>"
                    f"<td>{q.wall_seconds:.3f}s</td>"
                    f"<td>{q.eval_count}</td>"
                    f"<td>{q.tps:.2f}</td>"
                    "</tr>")
            else:
                parts.append(
                    f"<tr><td>{i}</td>"
                    f"<td class='prompt'>{prompt_cell}</td>"
                    f"<td colspan='6' class='err'>"
                    f"{_h(q.error or 'unknown error')}</td>"
                    "</tr>")
        parts.append("</tbody></table>")

    parts.append("</body></html>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Step C3 — email send (only when config.email is populated)
# ---------------------------------------------------------------------------


def send_summary_email(cfg: BenchConfig, outcomes: list[BenchOutcome],
                       *, stamp: str) -> None:
    """Ship the HTML body + JSON attachment via SMTP. No-op when
    ``cfg.email`` is None (script just printed the summary).

    Wrapped so a SMTP failure here cannot lose the stdout / JSON
    artifacts that the operator might want to grep later.
    """
    if cfg.email is None:
        log.info("config.email not set — skipping email")
        return
    html = render_email_html(outcomes, cfg, stamp=stamp)
    json_dump = render_json_dump(outcomes, cfg, stamp=stamp)
    try:
        send_email(
            cfg.email, html, json_dump,
            stamp=stamp,
            json_filename=f"ollama_multi_bench_{stamp}.json",
        )
    except Exception:
        log.exception("email send failed")


# ---------------------------------------------------------------------------
# Teardown (optional)
# ---------------------------------------------------------------------------


def uninstall_all(cfg: BenchConfig) -> None:
    """Best-effort serial uninstall when ``uninstall_after: true``."""
    for t in cfg.targets:
        try:
            log.info("[uninstall] %s ...", t.app_name)
            market_uninstall(t.app_name,
                             watch_minutes=cfg.uninstall_minutes,
                             delete_data=cfg.delete_data)
            log.info("[uninstall] %s done", t.app_name)
        except Exception as exc:
            log.warning("[uninstall] %s failed (continuing): %s",
                        t.app_name, exc)


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Multi-model ollama install + sequential multi-round "
                    "prompt + TTFT report.",
    )
    ap.add_argument("-c", "--config", required=True,
                    help="path to the JSON config file")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.log_file)
    set_cli_path(cfg.cli_path or DEFAULT_OLARES_CLI)

    log.info("loaded config %s: %d target(s), %d prompt(s) "
             "(email %s)",
             args.config, len(cfg.targets), len(cfg.prompts),
             "configured" if cfg.email else "disabled")

    stamp = utc_now_naive().strftime("%Y%m%d_%H%M%S")

    installs = install_sequential(cfg)
    # Build one outcome per target. Install failures still produce an
    # outcome (with install_error populated) so the summary + email show
    # the failure reason instead of silently skipping the row.
    outcomes: list[BenchOutcome] = []
    for inst in installs:
        outcome = BenchOutcome(
            target=inst.target,
            install_decision=inst.decision,
            install_duration_seconds=inst.duration_seconds,
            install_already_existed=inst.already_existed,
            install_error=inst.error,
        )
        if inst.error:
            log.warning("[bench] skipping %s prompts: install failed",
                        inst.target.app_name)
            outcomes.append(outcome)
            continue
        outcomes.append(bench_one_app(outcome, cfg))

    print_summary(outcomes)
    send_summary_email(cfg, outcomes, stamp=stamp)

    if cfg.uninstall_after:
        uninstall_all(cfg)

    install_failed = any(o.install_error for o in outcomes)
    bench_failed = any(o.error or o.any_prompt_failed
                       for o in outcomes if o.install_ok)
    if install_failed:
        return 1
    if bench_failed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
