"""Serialize a list of ModelResult to JSON + HTML on disk."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict

from llm_bench.constants import LOG_NAMESPACE
from llm_bench.data.html_report import render_html
from llm_bench.domain import ModelResult
from llm_bench.utils.time_utils import utc_now_naive

log = logging.getLogger(LOG_NAMESPACE)


def write_reports(results: list[ModelResult],
                  out_dir: str) -> tuple[str, str, str, str]:
    """Write JSON + HTML reports under `out_dir`. Returns
    (json_path, html_path, json_dump_string, html_string) so the caller
    can attach the same bytes to email without re-serializing.
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

    log.info("wrote %s and %s", json_path, html_path)
    return json_path, html_path, json_dump, html
