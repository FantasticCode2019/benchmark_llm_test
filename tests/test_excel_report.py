"""Smoke tests for ``llm_bench.data.excel_report``.

The renderer is pure data-in / bytes-out, so we round-trip the bytes
back through ``openpyxl`` and verify a few representative cells. We
cover four cases:

* mixed Ollama + OpenAI results -> only Ollama rows appear, one
  worksheet per configured prompt
* no Ollama results              -> empty workbook (filename=None,
                                    bytes=b"")
* Ollama row without a descriptor -> probe / capability columns
                                     render as "" without crashing
* model that never reached the prompt loop  -> still emitted as a row
                                               with blank timings and
                                               the model error in the
                                               Error column
"""
from __future__ import annotations

import io

from openpyxl import load_workbook

from llm_bench.data.excel_report import render_ollama_excel
from llm_bench.domain import ApiType, ModelResult, QuestionResult


def _make_ollama_result(*, with_descriptor: bool = True) -> ModelResult:
    """Two-prompt ollama result with one OK row and one failing row,
    so the renderer has both shapes to project onto its per-prompt
    sheets.
    """
    return ModelResult(
        app_name="ollama-qwen3",
        model="qwen3:8b",
        api_type=ApiType.OLLAMA,
        started_at="2026-05-12T10:00:00Z",
        finished_at="2026-05-12T10:02:00Z",
        endpoint="https://ollama.example/api",
        install_seconds=42.0,
        uninstall_seconds=7.0,
        ollama_supports_thinking=True,
        ollama_descriptor={
            "model": "qwen3:8b",
            "family": "qwen3",
            "parameter_size": "8.2B",
            "quantization": "Q4_K_M",
            "max_context": 40960,
            "runtime_context": 4096,
            "disk_gb": 4.5,
            "total_gb": 5.1,
            "vram_gb": 5.1,
            "ram_gb": 0.0,
            "kvcache_gb": 0.6,
            "processor": "100% GPU",
            "loaded": True,
        } if with_descriptor else None,
        questions=[
            QuestionResult(
                prompt="hi",
                ok=True,
                wall_seconds=1.0,
                ttft_seconds=0.2,
                thinking_ttft_seconds=0.05,
                eval_count=120,
                tps=85.0,
                has_thinking=True,
                total_server_seconds=0.95,
            ),
            QuestionResult(prompt="boom", ok=False, error="net"),
        ],
    )


def _make_openai_result() -> ModelResult:
    """vLLM-served openai-compatible result. Should be filtered out."""
    return ModelResult(
        app_name="vllm-llama",
        model="meta-llama/Llama-3-8B",
        api_type=ApiType.OPENAI,
        questions=[QuestionResult(prompt="x", ok=True)],
    )


def _load_sheet(payload: bytes, sheet_name: str | None = None) -> list[list]:
    """Round-trip the rendered bytes back through openpyxl so we can
    assert on cell values without depending on private internals.
    Defaults to the first sheet when ``sheet_name`` is omitted.
    """
    wb = load_workbook(io.BytesIO(payload), read_only=True)
    ws = wb[sheet_name] if sheet_name else wb.worksheets[0]
    return [list(row) for row in ws.iter_rows(values_only=True)]


def _sheet_names(payload: bytes) -> list[str]:
    wb = load_workbook(io.BytesIO(payload), read_only=True)
    return list(wb.sheetnames)


class TestRenderOllamaExcel:
    def test_empty_when_no_ollama_results(self) -> None:
        # OpenAI-only run should produce no workbook at all.
        filename, payload = render_ollama_excel([_make_openai_result()])
        assert filename is None
        assert payload == b""

    def test_one_sheet_per_prompt(self) -> None:
        # Mixed list — only the Ollama row survives, and we get one
        # sheet per configured prompt rather than a single averaged row.
        filename, payload = render_ollama_excel([
            _make_openai_result(),
            _make_ollama_result(),
        ])
        assert filename == "llm_bench_ollama.xlsx"

        # Two prompts in the result -> two sheets, named in order.
        assert _sheet_names(payload) == ["Prompt 1", "Prompt 2"]

        # First sheet: row 1 banner with the prompt text, row 2 header,
        # row 3+ one row per ollama model.
        rows = _load_sheet(payload, "Prompt 1")
        assert rows[0][0].startswith("Prompt 1: hi")
        header = rows[1]
        assert "App" in header and "Family" in header
        # Per-prompt columns replaced the old averaging columns.
        assert "TTFT (s)" in header
        assert "TPS" in header
        assert "Tokens" in header
        assert "Wall (s)" in header
        assert "OK" in header
        assert "Avg TTFT (s)" not in header
        assert "Avg TPS" not in header
        assert "Prompts OK" not in header

        ollama_row = dict(zip(header, rows[2], strict=True))
        assert ollama_row["App"] == "ollama-qwen3"
        assert ollama_row["Model"] == "qwen3:8b"
        assert ollama_row["API"] == "ollama"
        # Runtime probe is the sole thinking signal in the workbook.
        assert ollama_row["Supports Thinking"] == "Yes"
        # Per-prompt cells reflect the OK QuestionResult, NOT averages.
        assert ollama_row["OK"] == "Yes"
        assert ollama_row["TTFT (s)"] == 0.2
        assert ollama_row["Think TTFT (s)"] == 0.05
        assert ollama_row["TPS"] == 85.0
        assert ollama_row["Tokens"] == 120
        assert ollama_row["Wall (s)"] == 1.0
        assert ollama_row["Server (s)"] == 0.95
        # Descriptor fields are still duplicated on every prompt sheet
        # so each tab is self-contained.
        assert ollama_row["Family"] == "qwen3"
        assert ollama_row["Parameter Size"] == "8.2B"
        assert ollama_row["Quantization"] == "Q4_K_M"
        assert ollama_row["Max Context"] == 40960
        assert ollama_row["Processor Split"] == "100% GPU"
        assert ollama_row["Loaded"] is True

        # Second sheet: failing prompt — OK=No, error surfaced, blank
        # timing cells (the QuestionResult had ok=False / error="net").
        rows2 = _load_sheet(payload, "Prompt 2")
        assert rows2[0][0].startswith("Prompt 2: boom")
        header2 = rows2[1]
        ollama_row2 = dict(zip(header2, rows2[2], strict=True))
        assert ollama_row2["OK"] == "No"
        assert ollama_row2["Error"] == "net"
        # The QuestionResult exists with the default zero timings, so
        # the renderer fills 0.0 / 0 — NOT the "did not reach this
        # prompt" placeholder which is reserved for the truly missing
        # case (see test_missing_prompt_row).
        assert ollama_row2["TTFT (s)"] == 0.0
        assert ollama_row2["Tokens"] == 0

    def test_descriptor_absent_renders_blank_cells(self) -> None:
        result = _make_ollama_result(with_descriptor=False)
        filename, payload = render_ollama_excel([result])
        assert filename is not None
        rows = _load_sheet(payload, "Prompt 1")
        header = rows[1]
        ollama_row = dict(zip(header, rows[2], strict=True))
        # Descriptor-derived fields fall back to empty cells (None ->
        # blank in openpyxl) but the row itself is still emitted.
        assert ollama_row["Family"] in (None, "")
        assert ollama_row["Parameter Size"] in (None, "")
        assert ollama_row["Loaded"] in (None, "")
        # Per-prompt timing cells still come from the QuestionResult.
        assert ollama_row["OK"] == "Yes"
        assert ollama_row["TPS"] == 85.0

    def test_missing_prompt_row(self) -> None:
        """Model that failed before reaching prompt 2 still appears in
        the prompt-2 sheet with blank timings + the model-level error.
        """
        ok_model = _make_ollama_result()
        # Second model: install failed, so questions[] is empty and the
        # prompt-2 sheet should still emit a row with the model error.
        broken = ModelResult(
            app_name="ollama-broken",
            model="broken:1b",
            api_type=ApiType.OLLAMA,
            error="install failed",
        )
        filename, payload = render_ollama_excel([ok_model, broken])
        assert filename is not None
        # Two prompts come from ok_model.questions (the longest list).
        assert _sheet_names(payload) == ["Prompt 1", "Prompt 2"]

        rows = _load_sheet(payload, "Prompt 2")
        header = rows[1]
        # Two model rows on every sheet.
        assert len(rows) == 4  # banner + header + 2 model rows
        broken_row = dict(zip(header, rows[3], strict=True))
        assert broken_row["App"] == "ollama-broken"
        assert broken_row["OK"] == "No"
        # Model-level error surfaces because there is no QuestionResult.
        assert broken_row["Error"] == "install failed"
        # Timing cells are blank (string ""), not 0, so the operator
        # can tell "didn't run" apart from "ran and reported zero".
        assert broken_row["TTFT (s)"] in (None, "")
        assert broken_row["TPS"] in (None, "")
