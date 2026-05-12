"""Serialize a list of ModelResult to JSON + HTML (+ Ollama .xlsx) on disk."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from typing import NamedTuple

from llm_bench.constants import LOG_NAMESPACE
from llm_bench.data.excel_report import render_ollama_excel
from llm_bench.data.html_report import render_html
from llm_bench.domain import ModelResult
from llm_bench.utils.time_utils import utc_now_naive

log = logging.getLogger(LOG_NAMESPACE)


class ReportArtifacts(NamedTuple):
    """In-memory + on-disk handles for the artifacts produced by one
    run, so the SMTP layer can attach the same bytes we wrote to disk
    without re-serializing.

    ``excel_path`` / ``excel_bytes`` are populated only when the run
    contained at least one Ollama model; otherwise both are None / b""
    and the mailer drops the attachment.
    """
    json_path: str
    html_path: str
    json_dump: str
    html: str
    excel_path: str | None
    excel_bytes: bytes


def write_reports(results: list[ModelResult],
                  out_dir: str) -> ReportArtifacts:
    """Write JSON + HTML (+ Ollama .xlsx) reports under ``out_dir``.

    Returns a :class:`ReportArtifacts` named tuple so callers (cli +
    mailer) can attach the exact same bytes that hit disk without
    re-rendering. The .xlsx is Ollama-only by design — see
    :mod:`llm_bench.data.excel_report` for the column contract.
    """
    os.makedirs(out_dir, exist_ok=True)
    stamp = utc_now_naive().strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(out_dir, f"llm_bench_{stamp}.json")
    html_path = os.path.join(out_dir, f"llm_bench_{stamp}.html")

    json_dump = json.dumps([asdict(r) for r in results],
                           ensure_ascii=False, indent=2)
    with open(json_path, "w", encoding="utf-8") as fp:
        fp.write(json_dump)

    html = render_html(results)
    with open(html_path, "w", encoding="utf-8") as fp:
        fp.write(html)

    excel_path: str | None = None
    excel_filename, excel_bytes = render_ollama_excel(results)
    if excel_filename and excel_bytes:
        excel_path = os.path.join(out_dir, f"llm_bench_{stamp}.xlsx")
        with open(excel_path, "wb") as fp:
            fp.write(excel_bytes)
        log.info("wrote %s, %s and %s", json_path, html_path, excel_path)
    else:
        log.info("wrote %s and %s (no ollama rows; xlsx skipped)",
                 json_path, html_path)

    return ReportArtifacts(
        json_path=json_path,
        html_path=html_path,
        json_dump=json_dump,
        html=html,
        excel_path=excel_path,
        excel_bytes=excel_bytes,
    )
