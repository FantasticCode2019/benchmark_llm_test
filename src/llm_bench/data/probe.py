"""--probe: dump the env requirements of every chart in config.models
without installing or modifying anything.
"""
from __future__ import annotations

import sys

from llm_bench.utils.cli_runner import cli_json


def probe_apps(apps: list[str]) -> None:
    for app in apps:
        try:
            info = cli_json(["market", "get", app])
        except Exception as exc:
            print(f"{app}: probe failed -> {exc}", file=sys.stderr)
            continue
        envs = (info or {}).get("raw_data", {}).get("envs", [])
        title = (info or {}).get("app_info", {}).get("app_entry", {}).get("title")
        version = (info or {}).get("version", "")
        print(f"\n=== {app}  ({title or '-'}, v{version}) ===")
        if not envs:
            print("  no envs declared")
            continue
        for e in envs:
            name = e.get("envName", "?")
            required = "required" if e.get("required") else "optional"
            etype = e.get("type") or "-"
            value_from = (e.get("valueFrom") or {}).get("envName")
            default = e.get("default")
            line = f"  - {name}  ({required}, type={etype})"
            if value_from:
                line += f"  <- references USER env: {value_from}"
            elif default:
                line += f"  default={default!r}"
            print(line)
            desc = e.get("description")
            if desc:
                print(f"      desc: {desc}")
