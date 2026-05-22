"""Render the email-friendly per-prompt summary HTML.

The email body mirrors the Excel attachment's sheet model: one table
per configured prompt, with one row per model showing the
PER-PROMPT values (no averaging). Models that never reached a given
prompt (install / readiness failed, or a previous prompt error
short-circuited the loop) still appear in every prompt's table with
blank timing cells and a FAIL badge carrying the model-level error,
so the run roster stays complete.
"""
from __future__ import annotations

from html import escape as html_escape

from llm_bench.domain import ModelResult, QuestionResult
from llm_bench.utils.format import fmt_duration, preview_prompt
from llm_bench.utils.time_utils import beijing_now_naive


# ---------------------------------------------------------------------------
# Inline CSS snippets (Gmail strips <style> blocks, so everything is inline)
# ---------------------------------------------------------------------------

_TH = ("padding:8px 12px;background:#f5f6f8;"
       "border-bottom:1px solid #dcdfe4;"
       "color:#555;font-weight:600;font-size:11px;"
       "text-transform:uppercase;letter-spacing:.04em;"
       "white-space:nowrap;")
_TH_L = _TH + "text-align:left;"
_TH_C = _TH + "text-align:center;"
_TH_R = _TH + "text-align:right;"

_CELL = ("padding:9px 12px;border-bottom:1px solid #eef0f3;"
         "vertical-align:middle;font-size:13px;color:#222;")
_CELL_L = _CELL + "text-align:left;"
_CELL_C = _CELL + "text-align:center;"
_CELL_R = _CELL + ("text-align:right;font-variant-numeric:tabular-nums;"
                   "font-feature-settings:'tnum';")

_EMPTY = '<span style="color:#bbb">—</span>'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_prompts(results: list[ModelResult]) -> list[str]:
    """Recover the canonical configured prompt list from completed runs.

    Each model run records one :class:`QuestionResult` per prompt
    (the orchestrator appends one per iteration even when a single
    prompt errors), so the LONGEST list across all results IS the
    configured prompt list — order preserved because the
    orchestrator iterates the original list in order.

    Mirrors :func:`llm_bench.data.excel_report._collect_prompts` so
    the email body and the .xlsx attachment cover the same set of
    prompts in the same order.
    """
    best: list[str] = []
    for r in results:
        if len(r.questions) > len(best):
            best = [q.prompt for q in r.questions]
    return best


def _question_for(result: ModelResult,
                  prompt_idx: int) -> QuestionResult | None:
    """Return ``result.questions[prompt_idx]`` if it exists, else None.

    None happens when the model failed before reaching this prompt
    (install / readiness raised, so the orchestrator never appended a
    QuestionResult for this index). The renderer treats that as
    "blank timing cells + model-level error badge".
    """
    if 0 <= prompt_idx < len(result.questions):
        return result.questions[prompt_idx]
    return None


def _fail_badge(error: str) -> str:
    """Red FAIL badge with the full error text as a tooltip."""
    return ('<span style="display:inline-block;padding:2px 9px;'
            'border-radius:10px;background:#fdecea;color:#c0392b;'
            'font-size:11px;font-weight:600;letter-spacing:.02em" '
            f'title="{html_escape(error)}">FAIL</span>')


_OK_BADGE = ('<span style="display:inline-block;padding:2px 9px;'
             'border-radius:10px;background:#e6f7ee;color:#0a7d48;'
             'font-size:11px;font-weight:600;letter-spacing:.02em">'
             'OK</span>')


def _pipeline_subtitle(r: ModelResult) -> str:
    """Compact install/uninstall overhead string shown under each
    app name — keeps install/uninstall visibility without taking a
    column from the per-prompt table.
    """
    decision = r.install_decision or "-"
    install_s = r.install_seconds or 0
    if r.uninstall_skipped:
        tail = "uninstall skipped"
    elif r.uninstall_seconds:
        tail = f"uninstall {r.uninstall_seconds:.0f}s"
    else:
        tail = "uninstall n/a"
    return (f"{html_escape(str(decision))} · install {install_s:.0f}s · "
            f"{tail}")


def _logs_line(r: ModelResult) -> str:
    """Red "pod logs: <path>" line shown under the app name when the
    orchestrator successfully archived the pod logs (i.e. the run
    failed). Empty string when there is no archive.
    """
    if not r.pod_logs_archive:
        return ""
    return (
        '<div style="color:#c0392b;font-size:11px;margin-top:2px">'
        'pod logs: '
        f'<code style="background:#fdecea;padding:0 4px;'
        f'border-radius:3px;font-family:SFMono-Regular,Consolas,'
        f'Menlo,monospace">{html_escape(r.pod_logs_archive)}</code>'
        '</div>'
    )


def _model_row(r: ModelResult, prompt_idx: int, *, bg: str) -> str:
    """Render one ``<tr>`` for ``r`` on prompt index ``prompt_idx``.

    Cells reflect ONLY this prompt's :class:`QuestionResult` — no
    averaging. When the model never reached this prompt the timing
    cells stay blank and the Status column carries a FAIL badge with
    the model-level error.
    """
    qr = _question_for(r, prompt_idx)

    if qr is None:
        # Model failed before reaching this prompt — blank cells +
        # model-level error in the badge tooltip.
        ttft_think = _EMPTY
        ttft = _EMPTY
        tps = _EMPTY
        tokens = _EMPTY
        wall = _EMPTY
        badge = _fail_badge(r.error or "did not reach this prompt")
    elif qr.ok:
        ttft_think = (f"{qr.thinking_ttft_seconds:.2f}"
                      if qr.thinking_ttft_seconds else _EMPTY)
        ttft = f"{qr.ttft_seconds:.2f}"
        tps = f"{qr.tps:.1f}"
        tokens = f"{qr.eval_count:d}"
        wall = f"{qr.wall_seconds:.2f}"
        badge = _OK_BADGE
    else:
        # Prompt was attempted but errored — surface the per-prompt
        # error on the badge tooltip; numerics stay blank because the
        # request never produced a useful timing.
        ttft_think = _EMPTY
        ttft = _EMPTY
        tps = _EMPTY
        tokens = _EMPTY
        wall = _EMPTY
        badge = _fail_badge(qr.error or r.error or "prompt failed")

    return (
        f'<tr style="background:{bg};">'
        f'<td style="{_CELL_L}">'
        f'<div style="font-weight:600;color:#111">'
        f'{html_escape(r.app_name)}</div>'
        f'<div style="color:#888;font-size:11px;margin-top:2px">'
        f'{_pipeline_subtitle(r)}</div>'
        f'{_logs_line(r)}'
        f'</td>'
        f'<td style="{_CELL_L}">'
        f'<code style="background:#f3f4f6;padding:1px 6px;'
        f'border-radius:3px;font-size:12px;color:#1a1a1a;'
        f'font-family:SFMono-Regular,Consolas,Menlo,monospace">'
        f'{html_escape(r.model)}</code></td>'
        f'<td style="{_CELL_R}">{ttft_think}</td>'
        f'<td style="{_CELL_R}">{ttft}</td>'
        f'<td style="{_CELL_R};font-weight:600;color:#0b5fff">{tps}</td>'
        f'<td style="{_CELL_R}">{tokens}</td>'
        f'<td style="{_CELL_R}">{wall}</td>'
        f'<td style="{_CELL_C}">{badge}</td>'
        '</tr>'
    )


def _prompt_section(results: list[ModelResult], prompt_idx: int,
                    prompt: str) -> str:
    """Render the full per-prompt block: banner + table of every model
    on this prompt.

    Models are emitted in the caller-provided order; rows alternate
    background colour for legibility. When a model has no
    QuestionResult for this index (install/readiness failed), it
    still gets a row with blank timings — see :func:`_model_row`.
    """
    rows: list[str] = []
    for i, r in enumerate(results):
        bg = "#ffffff" if i % 2 == 0 else "#fafbfc"
        rows.append(_model_row(r, prompt_idx, bg=bg))

    banner_label = f"Prompt {prompt_idx + 1}"
    prompt_preview = preview_prompt(prompt)
    return (
        '<div style="margin:18px 0 0 0">'
        # Prompt banner — sized like a small section heading so the
        # eye can scan from one prompt to the next.
        '<div style="margin:0 0 6px 0;padding:8px 12px;'
        'background:#eaf1ff;border-radius:6px;'
        'font-size:13px;color:#111;line-height:1.45">'
        f'<span style="font-weight:700">{html_escape(banner_label)}:</span> '
        f'<span style="color:#333">{html_escape(prompt_preview)}</span>'
        '</div>'
        # Per-prompt table.
        '<div style="border:1px solid #e5e7eb;border-radius:8px;'
        'overflow:hidden;background:#fff">'
        '<table style="border-collapse:collapse;width:100%;'
        'font-size:13px"><thead><tr>'
        f'<th style="{_TH_L}">App</th>'
        f'<th style="{_TH_L}">Model</th>'
        f'<th style="{_TH_R}">Think TTFT (s)</th>'
        f'<th style="{_TH_R}">TTFT (s)</th>'
        f'<th style="{_TH_R}">TPS</th>'
        f'<th style="{_TH_R}">Tokens</th>'
        f'<th style="{_TH_R}">Wall (s)</th>'
        f'<th style="{_TH_C}">Status</th>'
        '</tr></thead><tbody>'
        + "".join(rows) +
        '</tbody></table></div>'
        '</div>'
    )


def _summary_stats(results: list[ModelResult]) -> str:
    """Header strapline: model count, prompt OK ratio, total wall."""
    total_models = len(results)
    ok_prompts = sum(sum(1 for q in r.questions if q.ok) for r in results)
    total_prompts = sum(len(r.questions) for r in results)
    failed_models = sum(
        1 for r in results
        if r.error is not None or not any(q.ok for q in r.questions)
    )
    total_wall = sum(
        q.wall_seconds for r in results for q in r.questions if q.ok
    )

    bits = [
        f'{total_models}&nbsp;model{"s" if total_models != 1 else ""}',
        f'{ok_prompts}/{total_prompts}&nbsp;prompts&nbsp;OK',
        f'wall&nbsp;{fmt_duration(total_wall)}',
    ]
    if failed_models:
        bits.append(
            f'<span style="color:#c0392b">'
            f'{failed_models}&nbsp;failed</span>'
        )
    return " &middot; ".join(bits)


def _empty_body(subtitle: str) -> str:
    """Body shown when the run completed without producing any
    prompts (every model failed before the prompt loop). We still
    emit the header so the operator knows something landed in their
    inbox; the JSON / pod-log attachment carries the failure
    details.
    """
    return (
        '<div style="font-family:-apple-system,BlinkMacSystemFont,'
        '\'Segoe UI\',Roboto,Helvetica,Arial,sans-serif;color:#222;'
        'max-width:920px;padding:16px 4px;line-height:1.45">'
        '<div style="margin:0 0 14px 0">'
        '<div style="font-size:20px;font-weight:600;color:#111;'
        'margin-bottom:2px">Olares LLM benchmark</div>'
        f'<div style="color:#666;font-size:13px">'
        f'{beijing_now_naive().strftime("%Y-%m-%d %H:%M 北京时间")}'
        f' &middot; {subtitle}</div>'
        '</div>'
        '<div style="border:1px solid #e5e7eb;border-radius:8px;'
        'padding:16px;background:#fff;color:#666;font-size:13px">'
        'No prompts were attempted in this run — every model failed '
        'before reaching the prompt loop. See the attached JSON / '
        'pod-log archive for details.'
        '</div>'
        '</div>'
    )


def _footer_legend() -> str:
    """Compact column-meaning footer; intentionally identical wording
    to the previous single-table layout so readers familiar with the
    legend don't have to re-learn it.
    """
    return (
        '<p style="margin:14px 2px 0;color:#888;font-size:11.5px;'
        'line-height:1.55">'
        'One table per prompt &middot; each row shows the model\'s '
        'per-prompt values (no averaging). '
        '<b>Think TTFT</b> = time to the model\'s FIRST '
        'reasoning/thinking token (Ollama <code>message.thinking</code>, '
        'vLLM <code>delta.reasoning</code>); empty '
        '(<span style="color:#bbb">—</span>) when the model has no '
        'thinking phase or the streaming probe failed &middot; '
        '<b>TTFT</b> = time to the first ANSWER token (after thinking, '
        'if any) &middot; '
        '<b>TPS</b> = generated tokens per second &middot; '
        '<b>Tokens</b> = generated tokens for this prompt &middot; '
        '<b>Wall</b> = client-side request &rarr; response. '
        'For Ollama these come from server-reported '
        '<code style="background:#f3f4f6;padding:0 4px;'
        'border-radius:3px">load/prompt_eval/eval</code> durations; '
        'for vLLM / llama.cpp TTFT is taken from the first '
        '<code>delta.content</code> of the streaming probe '
        '(max_tokens=1 fallback) and TPS prefers llama.cpp '
        '<code style="background:#f3f4f6;padding:0 4px;'
        'border-radius:3px">timings.predicted_per_second</code> '
        'when present, else completion_tokens / wall. '
        'API type, descriptor metadata, and per-prompt error text '
        'are preserved in the JSON / Excel attachments.'
        '</p>'
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_html(results: list[ModelResult]) -> str:
    """Email-friendly multi-table summary — one table per prompt.

    Layout mirrors :mod:`llm_bench.data.excel_report` (one sheet per
    prompt) so the email body and the .xlsx attachment present the
    same per-prompt numbers in the same order.

    Returns the empty-state body (no per-prompt tables) when every
    model failed before the prompt loop; the JSON / pod-log
    attachments still carry the failure detail in that case.
    """
    subtitle = _summary_stats(results)
    prompts = _collect_prompts(results)

    if not prompts:
        return _empty_body(subtitle)

    sections = [
        _prompt_section(results, idx, prompt)
        for idx, prompt in enumerate(prompts)
    ]

    return (
        '<div style="font-family:-apple-system,BlinkMacSystemFont,'
        '\'Segoe UI\',Roboto,Helvetica,Arial,sans-serif;color:#222;'
        'max-width:920px;padding:16px 4px;line-height:1.45">'
        # header card
        '<div style="margin:0 0 14px 0">'
        '<div style="font-size:20px;font-weight:600;color:#111;'
        'margin-bottom:2px">Olares LLM benchmark</div>'
        f'<div style="color:#666;font-size:13px">'
        f'{beijing_now_naive().strftime("%Y-%m-%d %H:%M 北京时间")}'
        f' &middot; {subtitle}</div>'
        '</div>'
        + "".join(sections)
        + _footer_legend()
        + '</div>'
    )


__all__ = ["render_html"]
