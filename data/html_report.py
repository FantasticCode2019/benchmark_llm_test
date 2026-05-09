"""Render the email-friendly summary table HTML."""
from __future__ import annotations

from datetime import datetime
from html import escape as html_escape

from utils.format import fmt_duration


def render_html(results: list) -> str:
    """Email-friendly single-table summary.

    Inlined CSS (Gmail strips <style> blocks). Numeric columns are
    right-aligned with tabular figures. Per-prompt detail lives in the
    attached JSON, not the email body.
    """
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

    th = ("padding:8px 12px;background:#f5f6f8;"
          "border-bottom:1px solid #dcdfe4;"
          "color:#555;font-weight:600;font-size:11px;"
          "text-transform:uppercase;letter-spacing:.04em;"
          "white-space:nowrap;")
    th_l = th + "text-align:left;"
    th_c = th + "text-align:center;"
    th_r = th + "text-align:right;"

    cell = ("padding:9px 12px;border-bottom:1px solid #eef0f3;"
            "vertical-align:middle;font-size:13px;color:#222;")
    cell_l = cell + "text-align:left;"
    cell_c = cell + "text-align:center;"
    cell_r = cell + ("text-align:right;font-variant-numeric:tabular-nums;"
                     "font-feature-settings:'tnum';")

    rows: list = []
    for idx, r in enumerate(results):
        ok_q = sum(1 for q in r.questions if q.ok)
        total_q = len(r.questions)
        run_ok = r.error is None and ok_q > 0
        bg = "#ffffff" if idx % 2 == 0 else "#fafbfc"

        if run_ok:
            badge = ('<span style="display:inline-block;padding:2px 9px;'
                     'border-radius:10px;background:#e6f7ee;color:#0a7d48;'
                     'font-size:11px;font-weight:600;letter-spacing:.02em">'
                     'OK</span>')
        else:
            err_full = r.error or "no successful prompt"
            badge = ('<span style="display:inline-block;padding:2px 9px;'
                     'border-radius:10px;background:#fdecea;color:#c0392b;'
                     'font-size:11px;font-weight:600;letter-spacing:.02em" '
                     f'title="{html_escape(err_full)}">FAIL</span>')

        # Install/uninstall pipeline overhead appears as a faint subtitle
        # under the app name — keeps the email "test parameters" focused
        # while still giving an at-a-glance install/cleanup signal.
        decision = r.install_decision or "-"
        install_s = r.install_seconds or 0
        if r.uninstall_skipped:
            tail = "uninstall skipped"
        elif r.uninstall_seconds:
            tail = f"uninstall {r.uninstall_seconds:.0f}s"
        else:
            tail = "uninstall n/a"
        sub = f"{html_escape(decision)} · install {install_s:.0f}s · {tail}"

        rows.append(
            f'<tr style="background:{bg};">'
            f'<td style="{cell_l}">'
            f'<div style="font-weight:600;color:#111">'
            f'{html_escape(r.app_name)}</div>'
            f'<div style="color:#888;font-size:11px;margin-top:2px">'
            f'{sub}</div>'
            f'</td>'
            f'<td style="{cell_l}">'
            f'<code style="background:#f3f4f6;padding:1px 6px;'
            f'border-radius:3px;font-size:12px;color:#1a1a1a;'
            f'font-family:SFMono-Regular,Consolas,Menlo,monospace">'
            f'{html_escape(r.model)}</code></td>'
            f'<td style="{cell_c};color:#666">{html_escape(r.api_type)}</td>'
            f'<td style="{cell_c}">{ok_q}/{total_q}</td>'
            f'<td style="{cell_r}">{r.avg("ttft_seconds"):.2f}</td>'
            f'<td style="{cell_r};font-weight:600;color:#0b5fff">'
            f'{r.avg("tps"):.1f}</td>'
            f'<td style="{cell_r}">{r.avg("eval_count"):.0f}</td>'
            f'<td style="{cell_r}">{r.avg("wall_seconds"):.2f}</td>'
            f'<td style="{cell_c}">{badge}</td>'
            "</tr>"
        )

    summary_bits = [
        f'{total_models}&nbsp;model{"s" if total_models != 1 else ""}',
        f'{ok_prompts}/{total_prompts}&nbsp;prompts&nbsp;OK',
        f'wall&nbsp;{fmt_duration(total_wall)}',
    ]
    if failed_models:
        summary_bits.append(
            f'<span style="color:#c0392b">'
            f'{failed_models}&nbsp;failed</span>'
        )
    subtitle = " &middot; ".join(summary_bits)

    return (
        '<div style="font-family:-apple-system,BlinkMacSystemFont,'
        '\'Segoe UI\',Roboto,Helvetica,Arial,sans-serif;color:#222;'
        'max-width:920px;padding:16px 4px;line-height:1.45">'
        # header card
        '<div style="margin:0 0 14px 0">'
        '<div style="font-size:20px;font-weight:600;color:#111;'
        'margin-bottom:2px">Olares LLM benchmark</div>'
        f'<div style="color:#666;font-size:13px">'
        f'{datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}'
        f' &middot; {subtitle}</div>'
        '</div>'
        # main table — wrapped in a div so the rounded border survives
        # email clients that drop border-radius on <table>.
        '<div style="border:1px solid #e5e7eb;border-radius:8px;'
        'overflow:hidden;background:#fff">'
        '<table style="border-collapse:collapse;width:100%;'
        'font-size:13px"><thead><tr>'
        f'<th style="{th_l}">App</th>'
        f'<th style="{th_l}">Model</th>'
        f'<th style="{th_c}">API</th>'
        f'<th style="{th_c}">OK / N</th>'
        f'<th style="{th_r}">TTFT (s)</th>'
        f'<th style="{th_r}">TPS</th>'
        f'<th style="{th_r}">Tokens</th>'
        f'<th style="{th_r}">Wall (s)</th>'
        f'<th style="{th_c}">Status</th>'
        '</tr></thead><tbody>'
        + "".join(rows) +
        '</tbody></table></div>'
        # footer legend — compact, single line wherever possible
        '<p style="margin:14px 2px 0;color:#888;font-size:11.5px;'
        'line-height:1.55">'
        '<b>TTFT</b> = time to first token &middot; '
        '<b>TPS</b> = generated tokens per second &middot; '
        '<b>Tokens</b> = avg generated tokens per prompt &middot; '
        '<b>Wall</b> = client-side request &rarr; response. '
        'Numbers are averaged over successful prompts. For Ollama these '
        'come from server-reported <code style="background:#f3f4f6;'
        'padding:0 4px;border-radius:3px">load/prompt_eval/eval</code> '
        'durations; for vLLM / llama.cpp under stream=false TTFT is '
        'approximated via a max_tokens=1 round-trip and TPS prefers '
        'llama.cpp <code style="background:#f3f4f6;padding:0 4px;'
        'border-radius:3px">timings.predicted_per_second</code> when '
        'present, else completion_tokens / wall. '
        'Per-prompt records, errors, and notes are in the attached JSON.'
        '</p>'
        '</div>'
    )
