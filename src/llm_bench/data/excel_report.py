"""Per-run Excel summary for Ollama models — one sheet per configured prompt.

The Excel attachment is INTENTIONALLY narrower than the JSON report:
it answers, *for each prompt*, "which models served it and how fast?"
vLLM / OpenAI rows do NOT appear here — the Excel exists so an
operator can scan a single grid of GPU/RAM/context / tokens-per-second
figures without filtering on api_type first.

Sheet model
-----------
* One worksheet per prompt configured in ``cfg.questions``. Sheets
  are titled ``Prompt 1``, ``Prompt 2``, … (Excel caps sheet names at
  31 chars, so the literal prompt text would not fit; the full prompt
  is shown as a banner on row 1 instead).
* Each sheet has one row per Ollama model. Cells carry the
  PER-PROMPT values (no averaging) — TTFT, Think TTFT, TPS, tokens,
  wall, server time, etc. — so the operator can compare every model
  on the same prompt side-by-side.
* Models that never reached the prompt loop (install / readiness
  failed) still appear so the run roster is complete; their timing
  cells are blank and the Error column carries the model-level
  failure reason.

Column layout (kept stable; mirrored by the column header row):

    App                     spec.app_name
    Model                   spec.model_name (server-discovered when applicable)
    API                     always "ollama" in this sheet
    Supports Thinking       runtime probe via /api/show capabilities[]
                            (also drives whether the streaming TTFT
                            probe runs — see orchestrator._run_one_prompt)
    Family                  /api/show details.family
    Parameter Size          /api/show details.parameter_size
    Quantization            /api/show details.quantization_level
    Max Context             /api/show model_info[*.context_length]
    Runtime Context         /api/ps entry.context_length
    Disk (GiB)              /api/tags entry.size / 1 GiB
    Total VRAM+RAM (GiB)    /api/ps entry.size / 1 GiB
    VRAM (GiB)              /api/ps entry.size_vram / 1 GiB
    RAM (GiB)               (size - size_vram) / 1 GiB
    KV Cache (GiB)          (size - disk) / 1 GiB (negative when unloaded)
    Processor Split         "100% GPU" / "X% GPU / Y% CPU" / "not loaded"
    Loaded                  bool — /api/ps had an entry for this model
    OK                      "Yes" / "No" — this prompt's QuestionResult.ok
    TTFT (s)                this prompt's QuestionResult.ttft_seconds
    Think TTFT (s)          this prompt's QuestionResult.thinking_ttft_seconds
    TPS                     this prompt's QuestionResult.tps
    Tokens                  this prompt's QuestionResult.eval_count
    Wall (s)                this prompt's QuestionResult.wall_seconds
    Server (s)              this prompt's QuestionResult.total_server_seconds
    Install Decision        "fresh" / "reused" / "recovered" / ""
    Install (s)             ModelResult.install_seconds
    Uninstall (s)           ModelResult.uninstall_seconds (0 if skipped)
    Started / Finished      UTC ISO timestamps from the run
    Endpoint                base URL the benchmark used
    Error                   per-prompt error if any, else model-level error

`render_ollama_excel` returns `(filename, bytes)` so callers can write
to disk AND attach to the email without re-serializing. When no
Ollama row is present (config has only OpenAI / vLLM models, or all
failed before any prompt ran), the function returns `(None, b"")` and
the orchestrator silently drops the attachment.
"""
from __future__ import annotations

import io
import logging
import re
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from llm_bench.constants import LOG_NAMESPACE
from llm_bench.domain import ApiType, ModelResult, QuestionResult

log = logging.getLogger(LOG_NAMESPACE)


_COLUMNS: list[tuple[str, str]] = [
    # (header, key in `_row_for`'s field_map)
    ("App",                  "app_name"),
    ("Model",                "model"),
    ("API",                  "api_type"),
    ("Supports Thinking",    "ollama_supports_thinking"),
    ("Family",               "family"),
    ("Parameter Size",       "parameter_size"),
    ("Quantization",         "quantization"),
    ("Max Context",          "max_context"),
    ("Runtime Context",      "runtime_context"),
    ("Disk (GiB)",           "disk_gb"),
    ("Total VRAM+RAM (GiB)", "total_gb"),
    ("VRAM (GiB)",           "vram_gb"),
    ("RAM (GiB)",            "ram_gb"),
    ("KV Cache (GiB)",       "kvcache_gb"),
    ("Processor Split",      "processor"),
    ("Loaded",               "loaded"),
    ("OK",                   "ok"),
    ("TTFT (s)",             "ttft"),
    ("Think TTFT (s)",       "thinking_ttft"),
    ("TPS",                  "tps"),
    ("Tokens",               "eval_count"),
    ("Wall (s)",             "wall"),
    ("Server (s)",           "total_server"),
    ("Install Decision",     "install_decision"),
    ("Install (s)",          "install_seconds"),
    ("Uninstall (s)",        "uninstall_seconds"),
    ("Started",              "started_at"),
    ("Finished",             "finished_at"),
    ("Endpoint",             "endpoint"),
    ("Error",                "error"),
]


# Excel sheet name constraints: max 31 chars, no `: \ / ? * [ ]`.
_INVALID_SHEET_CHARS = re.compile(r'[:\\/?*\[\]]')


def _safe_sheet_title(title: str) -> str:
    """Coerce ``title`` into something Excel accepts as a sheet name."""
    cleaned = _INVALID_SHEET_CHARS.sub("_", title).strip()
    return (cleaned or "Sheet")[:31]


def _descriptor_field(result: ModelResult, key: str) -> Any:
    """Pull `key` out of `result.ollama_descriptor`, tolerating the
    "descriptor wasn't recorded" case (orchestrator step skipped /
    daemon unreachable). Missing -> None, which renders as an empty
    cell in openpyxl.
    """
    d = result.ollama_descriptor
    if not isinstance(d, dict):
        return None
    return d.get(key)


def _format_tristate_bool(value: bool | None) -> str:
    """Render Optional[bool] as Yes / No / "" so empty doesn't look like
    a probe that explicitly returned False.
    """
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return ""


def _row_for(result: ModelResult, prompt_idx: int) -> list[Any]:
    """Project ``(result, prompt_idx)`` into the column order above.

    Per-prompt cells (TTFT, TPS, …) are filled from
    ``result.questions[prompt_idx]`` when that QuestionResult exists.
    When the model failed before reaching this prompt
    (install / readiness raised, so the orchestrator never appended a
    QuestionResult), the timing cells are left blank and the Error
    column carries ``result.error`` so the operator still sees why
    the row is empty.
    """
    qr: QuestionResult | None = (
        result.questions[prompt_idx]
        if 0 <= prompt_idx < len(result.questions)
        else None
    )

    if qr is None:
        ok_label = "No"
        ttft: Any = ""
        thinking_ttft: Any = ""
        tps: Any = ""
        eval_count: Any = ""
        wall: Any = ""
        total_server: Any = ""
        error_text = result.error or "did not reach this prompt"
    else:
        ok_label = "Yes" if qr.ok else "No"
        ttft = round(qr.ttft_seconds, 3)
        thinking_ttft = round(qr.thinking_ttft_seconds, 3)
        tps = round(qr.tps, 2)
        eval_count = qr.eval_count
        wall = round(qr.wall_seconds, 3)
        total_server = round(qr.total_server_seconds, 3)
        # Per-prompt error wins over model-level error (which may also
        # be set if a *later* prompt raised). When neither is set the
        # cell is empty.
        error_text = qr.error or result.error or ""

    field_map: dict[str, Any] = {
        "app_name": result.app_name,
        "model": result.model,
        "api_type": str(result.api_type),
        "ollama_supports_thinking": _format_tristate_bool(
            result.ollama_supports_thinking),
        "family": _descriptor_field(result, "family"),
        "parameter_size": _descriptor_field(result, "parameter_size"),
        "quantization": _descriptor_field(result, "quantization"),
        "max_context": _descriptor_field(result, "max_context"),
        "runtime_context": _descriptor_field(result, "runtime_context"),
        "disk_gb": _descriptor_field(result, "disk_gb"),
        "total_gb": _descriptor_field(result, "total_gb"),
        "vram_gb": _descriptor_field(result, "vram_gb"),
        "ram_gb": _descriptor_field(result, "ram_gb"),
        "kvcache_gb": _descriptor_field(result, "kvcache_gb"),
        "processor": _descriptor_field(result, "processor"),
        "loaded": _descriptor_field(result, "loaded"),
        "ok": ok_label,
        "ttft": ttft,
        "thinking_ttft": thinking_ttft,
        "tps": tps,
        "eval_count": eval_count,
        "wall": wall,
        "total_server": total_server,
        "install_decision": str(result.install_decision or ""),
        "install_seconds": result.install_seconds,
        "uninstall_seconds": result.uninstall_seconds,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "endpoint": result.endpoint,
        "error": error_text,
    }
    return [field_map[key] for _, key in _COLUMNS]


def _style_header(ws, *, row: int) -> None:
    """Bold + light-gray header row + sensible column widths so the
    sheet is legible without manual fiddling. Freeze panes are pinned
    just below the header.
    """
    header_font = Font(bold=True, color="333333")
    header_fill = PatternFill(start_color="F2F4F7",
                              end_color="F2F4F7",
                              fill_type="solid")
    header_align = Alignment(horizontal="left", vertical="center")
    for col_idx, (header, _) in enumerate(_COLUMNS, start=1):
        cell = ws.cell(row=row, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align
        # Width heuristic: 1.6x the header text, with a 12 / 42 floor + ceiling.
        ws.column_dimensions[get_column_letter(col_idx)].width = max(
            12, min(42, int(len(header) * 1.6) + 2))
    ws.freeze_panes = ws.cell(row=row + 1, column=1).coordinate


def _write_prompt_banner(ws, prompt_idx: int, prompt: str) -> None:
    """Write a merged banner on row 1 showing the full prompt text.

    Sheet titles are limited to 31 chars, so the literal prompt does
    not fit there; this banner is how the operator reads the actual
    question that produced the row data below.
    """
    banner = ws.cell(row=1, column=1,
                     value=f"Prompt {prompt_idx + 1}: {prompt}")
    banner.font = Font(bold=True, color="111111", size=12)
    banner.fill = PatternFill(start_color="EAF1FF",
                              end_color="EAF1FF",
                              fill_type="solid")
    banner.alignment = Alignment(horizontal="left", vertical="center",
                                 wrap_text=True)
    ws.merge_cells(start_row=1, start_column=1,
                   end_row=1, end_column=len(_COLUMNS))
    ws.row_dimensions[1].height = 28


def _collect_prompts(results: list[ModelResult]) -> list[str]:
    """Recover the canonical configured prompt sequence from the
    completed runs.

    Each model run records one :class:`QuestionResult` per prompt
    (the orchestrator appends one per iteration even when a single
    prompt errors), so the LONGEST list across all results IS the
    configured prompt list — order preserved because the orchestrator
    iterates the original list in order.

    When every model failed before reaching the prompt loop, returns
    an empty list and the caller drops the attachment.
    """
    best: list[str] = []
    for r in results:
        if len(r.questions) > len(best):
            best = [q.prompt for q in r.questions]
    return best


def render_ollama_excel(results: list[ModelResult],
                        ) -> tuple[str | None, bytes]:
    """Build the Ollama-only summary workbook with one sheet per prompt.

    Returns ``(filename_hint, content_bytes)``. The filename hint is
    the basename ``"llm_bench_ollama.xlsx"`` so the caller can decide
    where on disk to write it; bytes are the workbook content ready
    for both ``open(..., "wb").write(...)`` and SMTP attachment.

    Returns ``(None, b"")`` (and logs at INFO so the caller silently
    drops the attachment) in two cases:

      1. No Ollama models in the run (config has only OpenAI / vLLM).
      2. Ollama models present but every one of them failed before
         reaching the prompt loop, so there is nothing per-prompt to
         render.
    """
    ollama = [r for r in results if r.api_type == ApiType.OLLAMA]
    if not ollama:
        log.info("excel_report: no ollama results to write; "
                 "skipping .xlsx attachment")
        return None, b""

    prompts = _collect_prompts(ollama)
    if not prompts:
        log.info("excel_report: ollama models present but none reached the "
                 "prompt loop; skipping .xlsx attachment")
        return None, b""

    wb = Workbook()
    # Drop the default empty "Sheet" so we can populate ours fresh.
    default = wb.active
    wb.remove(default)

    header_row = 2
    data_start_row = header_row + 1

    for idx, prompt in enumerate(prompts):
        title = _safe_sheet_title(f"Prompt {idx + 1}")
        ws = wb.create_sheet(title=title)

        _write_prompt_banner(ws, idx, prompt)

        for col_idx, (header, _) in enumerate(_COLUMNS, start=1):
            ws.cell(row=header_row, column=col_idx, value=header)
        _style_header(ws, row=header_row)

        # Write data rows by explicit (row, col) so we are immune to
        # `Worksheet.append`'s reliance on the private `_current_row`
        # cursor (which is not advanced by direct ws.cell() writes).
        for r_offset, result in enumerate(ollama):
            for c_idx, value in enumerate(_row_for(result, idx), start=1):
                ws.cell(row=data_start_row + r_offset,
                        column=c_idx, value=value)

    buf = io.BytesIO()
    wb.save(buf)
    payload = buf.getvalue()
    log.info("excel_report: rendered %d prompt sheet(s) x %d ollama "
             "row(s) each, %d bytes",
             len(prompts), len(ollama), len(payload))
    return "llm_bench_ollama.xlsx", payload


__all__ = ["render_ollama_excel"]
