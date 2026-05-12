"""Per-model orchestration: install -> entrance -> readiness -> warmup
-> per-prompt benchmark -> uninstall.

The top-level :func:`bench_model` is intentionally short: it builds a
:class:`BenchmarkContext`, runs the steps in order, and converts any
exception into a recorded error on the result. The actual work lives in
the ``_step_*`` helpers below — each owns one logical phase and is
small enough to test (or replace) in isolation.

Phase ordering and failure semantics::

       ┌────────────────── try ────────────────────┐
       │  _step_install            (raises -> stop)│
       │  _step_resolve_entrance   (raises -> stop)│
       │  _step_open_entrance      (raises -> stop)│
       │  _step_wait_ready         (raises -> stop)│
       │  _step_optional_pull      (warns; never raises)│
       │  _step_run_prompts        (per-prompt try; never raises overall)│
       └───────────────────────────────────────────┘
       ┌──────────────── finally ─────────────────┐
       │  _step_archive_pod_logs   (warns; never raises)│
       │  _step_uninstall          (records error if it fails)│
       │  finished_at stamp                        │
       └───────────────────────────────────────────┘

Each ``_step_*`` mutates ``ctx`` (and ``ctx.result``) in place; the
caller never has to inspect step return values.
"""
from __future__ import annotations

import logging
import subprocess
import time

from llm_bench.constants import LOG_NAMESPACE
from llm_bench.core._context import BenchmarkContext
from llm_bench.core.benchmark.ollama import benchmark_prompt_ollama, pull_model_ollama
from llm_bench.core.benchmark.openai import benchmark_prompt_openai, openai_config_from
from llm_bench.core.entrance import ensure_entrance_public, find_entrance
from llm_bench.core.lifecycle import archive_pod_logs, ensure_installed, market_uninstall
from llm_bench.core.readiness import wait_until_api_ready
from llm_bench.domain import (
    ApiType,
    AppConfig,
    ModelResult,
    ModelSpec,
    QuestionResult,
    ResolvedOptions,
)
from llm_bench.utils.cli_runner import cli
from llm_bench.utils.time_utils import utc_now_naive

log = logging.getLogger(LOG_NAMESPACE)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def bench_model(spec: ModelSpec, prompts: list[str],
                cfg: AppConfig) -> ModelResult:
    """Run the full benchmark for one ``spec`` and return its aggregated
    :class:`ModelResult`.

    Behaviour matches the v0.1 monolithic implementation byte-for-byte
    on the JSON wire format; the only thing that changed is the
    *internal* structure (8 small steps + a context object instead of a
    single 200-line function with 15 inline ``_opt`` lookups).
    """
    opts = ResolvedOptions.for_model(spec, cfg.defaults)
    ctx = BenchmarkContext(
        spec=spec,
        cfg=cfg,
        opts=opts,
        openai=openai_config_from(spec, cfg),
        result=ModelResult(app_name=spec.app_name, model=spec.model_name,
                           api_type=opts.api_type,
                           started_at=utc_now_naive().isoformat() + "Z"),
        model_name=spec.model_name,
        prompts=list(prompts),
    )

    try:
        _step_install(ctx)
        _step_resolve_entrance(ctx)
        _step_open_entrance(ctx)
        _step_wait_ready(ctx)
        _step_optional_pull(ctx)
        _step_run_prompts(ctx)
    except subprocess.CalledProcessError as exc:
        _record_cli_error(ctx, exc)
    except Exception as exc:  # any other failure is captured on the result
        log.exception("model %s failed", ctx.app)
        ctx.set_error(str(exc))

    finally:
        _step_archive_pod_logs(ctx, post_uninstall_failure=False)
        _step_uninstall(ctx)
        ctx.result.finished_at = utc_now_naive().isoformat() + "Z"

    return ctx.result


# ---------------------------------------------------------------------------
# Step 1 — install / reuse / recover
# ---------------------------------------------------------------------------


def _step_install(ctx: BenchmarkContext) -> None:
    """Wrap :func:`ensure_installed` with timing + result mutation.

    Raises on install failure — the orchestrator's outer try/except
    converts the exception to ``ctx.result.error``.
    """
    t = time.perf_counter()
    already_existed, decision = ensure_installed(
        ctx.app,
        install_minutes=ctx.opts.install_minutes,
        uninstall_minutes=ctx.opts.uninstall_minutes,
        install_envs=list(ctx.spec.envs),
        delete_data=ctx.opts.delete_data,
        skip_if_running=ctx.opts.skip_if_running,
    )
    ctx.already_existed = already_existed
    ctx.result.install_decision = decision
    ctx.result.install_seconds = round(time.perf_counter() - t, 1)
    ctx.result.install_ok = True


# ---------------------------------------------------------------------------
# Step 2 — discover the entrance / honor explicit override
# ---------------------------------------------------------------------------


def _step_resolve_entrance(ctx: BenchmarkContext) -> None:
    """Pick the entrance + base URL the rest of the pipeline will talk to.

    Stores the URL on ``ctx.result.endpoint`` immediately so it shows
    up in the JSON even if a later step fails.
    """
    entrance, url, auth_level = find_entrance(
        ctx.app, ctx.spec.entrance_name, override=ctx.spec.endpoint_url)
    ctx.entrance = entrance
    ctx.entrance_url = url
    ctx.auth_level = auth_level or ""
    ctx.result.endpoint = url
    log.info("using entrance %s -> %s (authLevel=%s)",
             entrance, url, auth_level or "n/a")


# ---------------------------------------------------------------------------
# Step 3 — flip the entrance to public if needed
# ---------------------------------------------------------------------------


def _step_open_entrance(ctx: BenchmarkContext) -> None:
    """Forward to :func:`ensure_entrance_public`.

    Kept as a one-liner step so the top-level orchestrator reads as a
    sequence of named phases; do not inline back into ``bench_model``.
    """
    ensure_entrance_public(ctx.app, ctx.entrance, ctx.auth_level,
                           auto_open=ctx.opts.auto_open)


# ---------------------------------------------------------------------------
# Step 4 — wait for the bundle to download + server to load
# ---------------------------------------------------------------------------


def _step_wait_ready(ctx: BenchmarkContext) -> None:
    """Poll until the chart's bundle + server are ready to serve.

    When the server reports a different model identifier (vLLM's
    ``/v1/models`` does this for some chart launchers), we switch
    ``ctx.model_name`` to the server-reported value so the benchmark
    uses the name the server actually accepts.
    """
    discovered = wait_until_api_ready(
        ctx.entrance_url, ctx.opts.api_type.value, ctx.model_name,
        max_wait_minutes=ctx.opts.api_ready_minutes,
    )
    if discovered and discovered != ctx.model_name:
        log.info("server reports served name as %r (configured %r); "
                 "using server name for the benchmark",
                 discovered, ctx.model_name)
        ctx.model_name = discovered
        ctx.result.model = discovered


# ---------------------------------------------------------------------------
# Step 5 — opt-in explicit ollama /api/pull
# ---------------------------------------------------------------------------


def _step_optional_pull(ctx: BenchmarkContext) -> None:
    """Run an explicit ``ollama /api/pull`` when requested.

    Disabled by default — every olares-market ``ollama*`` chart ships
    a launcher that already pulls. Failure is logged as a warning and
    never propagated: if the model is missing the readiness poll
    already failed earlier, and if it's present the pull is just noise.
    """
    if ctx.opts.api_type is not ApiType.OLLAMA or not ctx.opts.pull_model:
        return
    try:
        pull_model_ollama(
            ctx.entrance_url, ctx.model_name,
            timeout=ctx.opts.pull_timeout,
            max_attempts=ctx.opts.pull_max_attempts,
            retry_sleep_seconds=ctx.opts.pull_retry_sleep_seconds,
        )
    except Exception as exc:  # pull is opt-in; failures are non-fatal
        log.warning("pull failed (model already present, "
                    "ignoring): %s", exc)


# ---------------------------------------------------------------------------
# Step 6 — per-prompt benchmark
# ---------------------------------------------------------------------------


def _step_run_prompts(ctx: BenchmarkContext) -> None:
    """Loop over the prompts, dispatch to the right backend, log results.

    Per-prompt failures stay in the :class:`QuestionResult`; we don't
    raise here because one bad prompt shouldn't lose the data of the
    other prompts (the benchmark backends already wrap their own
    request-level exceptions into ``ok=False`` results).
    """
    for prompt in ctx.prompts:
        log.info("prompt: %s", prompt[:60].replace("\n", " "))
        qr = _run_one_prompt(ctx, prompt)
        ctx.result.questions.append(qr)
        _log_prompt_result(ctx.opts.api_type, qr)


def _run_one_prompt(ctx: BenchmarkContext, prompt: str) -> QuestionResult:
    """Dispatch a single prompt to the correct backend."""
    if ctx.opts.api_type is ApiType.OPENAI:
        return benchmark_prompt_openai(
            ctx.entrance_url, ctx.model_name, prompt, ctx.openai,
            request_timeout=ctx.opts.request_timeout,
            thinking=ctx.opts.thinking,
        )
    return benchmark_prompt_ollama(
        ctx.entrance_url, ctx.model_name, prompt,
        request_timeout=ctx.opts.request_timeout,
        thinking=ctx.opts.thinking,
    )


def _log_prompt_result(api_type: ApiType, qr: QuestionResult) -> None:
    """Format the post-prompt log line. Ollama and OpenAI differ in
    which fields are meaningful, so the format strings split out.
    """
    if not qr.ok:
        log.warning("  -> error: %s", qr.error)
        return

    # Surface the "first thinking token" timing whenever it differs
    # from the answer-side TTFT (i.e. the model actually had a
    # reasoning phase to measure).
    think = ""
    if (qr.thinking_ttft_seconds
            and qr.thinking_ttft_seconds != qr.ttft_seconds):
        think = f" think_ttft={qr.thinking_ttft_seconds:.3f}s"

    if api_type is ApiType.OPENAI:
        log.info("  -> wall=%.3fs ttft~%.3fs%s tokens=%d "
                 "tps=%.2f (client_tps=%.2f, server_tps=%.2f)",
                 qr.wall_seconds, qr.ttft_seconds, think,
                 qr.eval_count, qr.tps, qr.client_tps,
                 qr.server_tps_reported)
    else:
        log.info("  -> ttft=%.3fs%s tokens=%d tps=%.2f wall=%.3fs",
                 qr.ttft_seconds, think, qr.eval_count,
                 qr.tps, qr.wall_seconds)


# ---------------------------------------------------------------------------
# Step 7 — pod log archive (finally-block)
# ---------------------------------------------------------------------------


def _step_archive_pod_logs(ctx: BenchmarkContext, *,
                           post_uninstall_failure: bool) -> None:
    """Archive ``/var/log/pods/*<app>*`` BEFORE uninstall (which deletes
    the pods + their log dirs). Two call sites:

      1. First-pass: from the outer finally, when anything went wrong
         in the try-block (install / readiness / every prompt failed).
      2. Second-pass: from inside :func:`_step_uninstall` when uninstall
         itself failed and the pods are still around — this is our last
         chance to grab the logs.

    Either is best-effort: failures here are warnings only so they
    don't shadow the actual run failure that put us here.
    """
    if not ctx.opts.save_pod_logs:
        return
    should_archive = (
        post_uninstall_failure and ctx.result.pod_logs_archive is None
    ) or (
        not post_uninstall_failure
        and (ctx.result.error is not None or not ctx.any_prompt_ok)
    )
    if not should_archive:
        return
    try:
        archive = archive_pod_logs(
            ctx.app, output_dir=ctx.opts.pod_logs_dir,
            sudo_password=ctx.cfg.sudo_password)
    except Exception as exc:  # pod-log archive is best-effort
        log.warning("%s: pod log archive raised: %s", ctx.app, exc)
        return
    if archive:
        ctx.result.pod_logs_archive = archive
        suffix = " (post uninstall failure)" if post_uninstall_failure else ""
        log.info("%s: pod logs saved to %s%s", ctx.app, archive, suffix)
        ctx.archived_logs = True


# ---------------------------------------------------------------------------
# Step 8 — uninstall (finally-block)
# ---------------------------------------------------------------------------


def _step_uninstall(ctx: BenchmarkContext) -> None:
    """Tear the chart back down per the user's policy.

    Priority: ``uninstall_after_run=false`` > ``preserve_if_existed +
    pre-existing`` > default(uninstall).
    """
    if not ctx.opts.uninstall_after:
        log.info("%s: uninstall_after_run=false; skipping post-benchmark "
                 "uninstall", ctx.app)
        ctx.result.uninstall_skipped = True
        ctx.result.uninstall_ok = True
        return

    if ctx.already_existed and ctx.opts.preserve_if_existed:
        log.info("%s was pre-existing and preserve_if_existed=true; "
                 "skipping post-benchmark uninstall", ctx.app)
        ctx.result.uninstall_skipped = True
        ctx.result.uninstall_ok = True
        return

    try:
        t = time.perf_counter()
        market_uninstall(ctx.app, watch_minutes=ctx.opts.uninstall_minutes,
                         delete_data=ctx.opts.delete_data)
        ctx.result.uninstall_seconds = round(time.perf_counter() - t, 1)
        ctx.result.uninstall_ok = True
    except Exception as exc:  # record + try one more log archive
        log.exception("uninstall %s failed", ctx.app)
        ctx.set_error(f"uninstall: {exc}")
        # Uninstall failed → pods are still around; if we didn't archive
        # in the first pass (because prompts succeeded), grab them now.
        _step_archive_pod_logs(ctx, post_uninstall_failure=True)


# ---------------------------------------------------------------------------
# Shared error helpers
# ---------------------------------------------------------------------------


def _record_cli_error(ctx: BenchmarkContext,
                      exc: subprocess.CalledProcessError) -> None:
    """Render an olares-cli failure with its captured stderr/stdout."""
    msg = (f"{cli()} failed (exit={exc.returncode}): "
           f"{(exc.stderr or exc.stdout or '').strip()[:500]}")
    log.error(msg)
    ctx.set_error(msg)


__all__ = ["bench_model"]
