"""Tests for the Phase-4 ``BenchmarkContext`` + orchestrator helper functions.

We can't exercise the full ``bench_model`` pipeline without a live
olares-cli + chart, but the small helpers around it (set_error, label
formatting, log dispatch) are pure and worth covering — they used to
be inlined in a 200-line function where they were hard to isolate.
"""
from __future__ import annotations

import logging

from llm_bench.core._context import BenchmarkContext
from llm_bench.core.orchestrator import (
    _log_prompt_result,
    _step_describe_ollama_model,
    _step_probe_ollama_thinking,
)
from llm_bench.domain import (
    ApiType,
    AppConfig,
    EmailConfig,
    GlobalDefaults,
    ModelResult,
    ModelSpec,
    OpenAIConfig,
    QuestionResult,
    ResolvedOptions,
)


def _make_ctx(*, error: str | None = None,
              questions: list[QuestionResult] | None = None,
              api_type: ApiType = ApiType.OLLAMA,
              ) -> BenchmarkContext:
    """Build a BenchmarkContext with minimal but valid wiring."""
    spec = ModelSpec(app_name="appX", model_name="mX", api_type=api_type)
    defaults = GlobalDefaults()
    cfg = AppConfig(
        defaults=defaults,
        models=[spec],
        questions=["q"],
        email=EmailConfig(
            smtp_host="h", smtp_port=587, username="u",
            password="p", sender="u@x", to="u@x",
        ),
    )
    ctx = BenchmarkContext(
        spec=spec,
        cfg=cfg,
        opts=ResolvedOptions.for_model(spec, defaults),
        openai=OpenAIConfig(),
        result=ModelResult(app_name="appX", model="mX",
                           api_type=api_type,
                           error=error,
                           questions=list(questions or [])),
        model_name="mX",
    )
    # Entrance URL is populated by `_step_resolve_entrance`; the two
    # ollama probe steps need it to call the client, so simulate that
    # having already happened.
    ctx.entrance_url = "http://ollama.local"
    return ctx


class TestBenchmarkContext:

    def test_app_property_aliases_app_name(self):
        ctx = _make_ctx()
        assert ctx.app == "appX"

    def test_set_error_records_first_failure_only(self):
        ctx = _make_ctx()
        ctx.set_error("first failure")
        ctx.set_error("second failure (cascaded)")
        # The first error stays; we don't overwrite with cascades.
        assert ctx.result.error == "first failure"

    def test_set_error_can_set_when_none(self):
        ctx = _make_ctx()
        assert ctx.result.error is None
        ctx.set_error("boom")
        assert ctx.result.error == "boom"

    def test_any_prompt_ok_false_for_empty(self):
        ctx = _make_ctx()
        assert ctx.any_prompt_ok is False

    def test_any_prompt_ok_false_for_all_failed(self):
        ctx = _make_ctx(questions=[
            QuestionResult(prompt="q1", ok=False),
            QuestionResult(prompt="q2", ok=False),
        ])
        assert ctx.any_prompt_ok is False

    def test_any_prompt_ok_true_when_one_succeeded(self):
        ctx = _make_ctx(questions=[
            QuestionResult(prompt="q1", ok=False),
            QuestionResult(prompt="q2", ok=True),
        ])
        assert ctx.any_prompt_ok is True


class TestLogPromptResult:
    """The pre-Phase-4 code inlined a 15-line if/else into the prompt
    loop; now it's a top-level helper. Verify the two backend branches
    log distinct shapes.
    """

    def test_failed_row_logs_warning(self, caplog):
        qr = QuestionResult(prompt="q", ok=False, error="boom")
        with caplog.at_level(logging.WARNING, logger="llm_bench"):
            _log_prompt_result(ApiType.OLLAMA, qr)
        assert any("boom" in rec.message and rec.levelname == "WARNING"
                   for rec in caplog.records)

    def test_ollama_ok_logs_ttft(self, caplog):
        qr = QuestionResult(prompt="q", ok=True,
                            ttft_seconds=0.5, eval_count=10,
                            tps=20.0, wall_seconds=1.2)
        with caplog.at_level(logging.INFO, logger="llm_bench"):
            _log_prompt_result(ApiType.OLLAMA, qr)
        msg = " ".join(rec.message for rec in caplog.records)
        # Ollama uses precise server-side ttft and never shows client_tps.
        assert "ttft=0.500s" in msg
        assert "client_tps" not in msg

    def test_openai_ok_logs_wall_and_client_tps(self, caplog):
        qr = QuestionResult(prompt="q", ok=True,
                            wall_seconds=2.0, ttft_seconds=0.3,
                            eval_count=100, tps=50.0,
                            client_tps=49.0, server_tps_reported=51.0)
        with caplog.at_level(logging.INFO, logger="llm_bench"):
            _log_prompt_result(ApiType.OPENAI, qr)
        msg = " ".join(rec.message for rec in caplog.records)
        # OpenAI surfaces both client_tps and server_tps to show the gap.
        assert "client_tps" in msg
        assert "server_tps" in msg

    def test_thinking_ttft_only_logged_when_distinct_from_ttft(self, caplog):
        """A reasoning model has thinking_ttft < ttft (reasoning emits
        first). The helper suppresses the `think_ttft=` log fragment
        when thinking_ttft equals ttft (i.e. nothing to add)."""
        qr_distinct = QuestionResult(prompt="q", ok=True,
                                     ttft_seconds=0.5,
                                     thinking_ttft_seconds=0.1,
                                     eval_count=5, tps=10.0,
                                     wall_seconds=1.0)
        with caplog.at_level(logging.INFO, logger="llm_bench"):
            _log_prompt_result(ApiType.OLLAMA, qr_distinct)
        assert any("think_ttft=" in rec.message for rec in caplog.records)

    def test_thinking_ttft_suppressed_when_equal(self, caplog):
        qr_equal = QuestionResult(prompt="q", ok=True,
                                  ttft_seconds=0.5,
                                  thinking_ttft_seconds=0.5,
                                  eval_count=5, tps=10.0,
                                  wall_seconds=1.0)
        with caplog.at_level(logging.INFO, logger="llm_bench"):
            _log_prompt_result(ApiType.OLLAMA, qr_equal)
        assert not any("think_ttft=" in rec.message for rec in caplog.records)


class TestOllamaSupportsThinkingStep:
    """``_step_probe_ollama_thinking`` only runs for ollama and records
    the probe result onto ``ctx.result.ollama_supports_thinking``.
    Failures stay best-effort: they leave the field at None.
    """

    def test_skipped_for_openai_api(self, monkeypatch):
        ctx = _make_ctx(api_type=ApiType.OPENAI)
        called = {"n": 0}

        def fake(*a, **kw):
            called["n"] += 1
            return True

        monkeypatch.setattr(
            "llm_bench.core.orchestrator.ollama_supports_thinking", fake)
        _step_probe_ollama_thinking(ctx)
        assert called["n"] == 0
        assert ctx.result.ollama_supports_thinking is None

    def test_records_true_for_ollama(self, monkeypatch):
        ctx = _make_ctx(api_type=ApiType.OLLAMA)
        monkeypatch.setattr(
            "llm_bench.core.orchestrator.ollama_supports_thinking",
            lambda *a, **kw: True)
        _step_probe_ollama_thinking(ctx)
        assert ctx.result.ollama_supports_thinking is True

    def test_records_false_for_ollama(self, monkeypatch):
        ctx = _make_ctx(api_type=ApiType.OLLAMA)
        monkeypatch.setattr(
            "llm_bench.core.orchestrator.ollama_supports_thinking",
            lambda *a, **kw: False)
        _step_probe_ollama_thinking(ctx)
        assert ctx.result.ollama_supports_thinking is False

    def test_swallows_exception(self, monkeypatch, caplog):
        ctx = _make_ctx(api_type=ApiType.OLLAMA)

        def boom(*a, **kw):
            raise RuntimeError("daemon down")

        monkeypatch.setattr(
            "llm_bench.core.orchestrator.ollama_supports_thinking", boom)
        with caplog.at_level(logging.WARNING, logger="llm_bench"):
            _step_probe_ollama_thinking(ctx)
        # Field remains None so downstream can tell "probe failed" from
        # "probe ran and returned False".
        assert ctx.result.ollama_supports_thinking is None
        assert any("ollama_supports_thinking" in rec.message
                   for rec in caplog.records)


class TestOllamaDescribeModelStep:
    """``_step_describe_ollama_model`` stores the descriptor dict for
    ollama runs and leaves non-ollama / failing runs untouched.
    """

    def test_skipped_for_openai_api(self, monkeypatch):
        ctx = _make_ctx(api_type=ApiType.OPENAI)
        called = {"n": 0}

        def fake(*a, **kw):
            called["n"] += 1
            return {"family": "x"}

        monkeypatch.setattr(
            "llm_bench.core.orchestrator.ollama_describe_model", fake)
        _step_describe_ollama_model(ctx)
        assert called["n"] == 0
        assert ctx.result.ollama_descriptor is None

    def test_records_descriptor_for_ollama(self, monkeypatch):
        ctx = _make_ctx(api_type=ApiType.OLLAMA)
        sample = {
            "model": "qwen3:8b",
            "family": "qwen3",
            "parameter_size": "8.2B",
            "quantization": "Q4_K_M",
            "max_context": 40960,
        }
        monkeypatch.setattr(
            "llm_bench.core.orchestrator.ollama_describe_model",
            lambda *a, **kw: sample)
        _step_describe_ollama_model(ctx)
        assert ctx.result.ollama_descriptor == sample

    def test_swallows_exception(self, monkeypatch, caplog):
        ctx = _make_ctx(api_type=ApiType.OLLAMA)

        def boom(*a, **kw):
            raise RuntimeError("describe failed")

        monkeypatch.setattr(
            "llm_bench.core.orchestrator.ollama_describe_model", boom)
        with caplog.at_level(logging.WARNING, logger="llm_bench"):
            _step_describe_ollama_model(ctx)
        assert ctx.result.ollama_descriptor is None
        assert any("ollama_describe_model" in rec.message
                   for rec in caplog.records)
