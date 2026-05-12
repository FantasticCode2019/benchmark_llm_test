"""Olares sequential LLM benchmark.

Top-level package. Public re-exports are intentionally kept minimal — the
canonical import surface is the sub-packages (`llm_bench.core`,
`llm_bench.data`, `llm_bench.clients`, `llm_bench.utils`,
`llm_bench.models`, `llm_bench.exceptions`).

Layout (src/ layout, see pyproject.toml):

    src/llm_bench/
        __init__.py              <-- this file
        __main__.py              <-- python -m llm_bench
        cli.py                   <-- argparse entrypoint + main()
        constants.py             <-- centralized magic numbers
        exceptions.py            <-- BenchmarkError hierarchy
        models.py                <-- result/config dataclasses
        core/                    <-- chart lifecycle + readiness + benchmark loop
        data/                    <-- config IO, report writer, mailer, probe
        clients/                 <-- ollama / vllm / openai-compat HTTP clients
        utils/                   <-- cli runner, HTTP base, format, token math
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
