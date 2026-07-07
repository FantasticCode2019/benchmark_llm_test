"""一次性 Ollama 基准测试脚本。

给定 entrance-url、model 和 prompt，调用 benchmark_prompt_ollama 发起
一次流式 /api/chat 请求，并把返回的 QuestionResult 的所有指标打印出来。

用法示例：

    python test_ollama.py \
        --entrance-url http://127.0.0.1:11434 \
        --model qwen3:8b \
        --prompt "用一句话介绍你自己" \
        --thinking

也可以走位置参数：

    python test.py http://127.0.0.1:11434 qwen3:8b "你好"
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

# 让脚本无需安装包即可直接运行：把 src/ 加入 import 搜索路径。
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm_bench.core.benchmark.ollama import benchmark_prompt_ollama  # noqa: E402
from llm_bench.domain import QuestionResult  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="给定 entrance-url / model / prompt，返回 QuestionResult 全部指标。",
    )
    parser.add_argument(
        "entrance_url",
        nargs="?",
        help="Ollama 入口地址，例如 http://127.0.0.1:11434",
    )
    parser.add_argument("model", nargs="?", help="模型 id，例如 qwen3:8b")
    parser.add_argument("prompt", nargs="?", help="要发送的提示词")

    parser.add_argument("--entrance-url", dest="entrance_url_opt", help="同位置参数 entrance_url")
    parser.add_argument("--model", dest="model_opt", help="同位置参数 model")
    parser.add_argument("--prompt", dest="prompt_opt", help="同位置参数 prompt")

    parser.add_argument(
        "--thinking",
        action="store_true",
        help="开启 thinking 计时（测量 thinking_ttft_seconds）。",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=300,
        help="单次请求读超时（秒），默认 300。",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="遇到瞬时网络错误时的最大尝试次数，默认 3。",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="仅以 JSON 形式输出全部指标。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印 benchmark 内部的 INFO 级日志。",
    )

    args = parser.parse_args(argv)

    # 选项形式优先覆盖位置参数。
    args.entrance_url = args.entrance_url_opt or args.entrance_url
    args.model = args.model_opt or args.model
    args.prompt = args.prompt_opt or args.prompt

    missing = [
        name
        for name, value in (
            ("entrance-url", args.entrance_url),
            ("model", args.model),
            ("prompt", args.prompt),
        )
        if not value
    ]
    if missing:
        parser.error(f"缺少必需参数: {', '.join(missing)}")

    return args


def print_result(result: QuestionResult) -> None:
    """以对齐的表格形式打印 QuestionResult 的全部字段。"""
    data = dataclasses.asdict(result)
    width = max(len(k) for k in data)
    print("\n===== QuestionResult 全部指标 =====")
    for key, value in data.items():
        print(f"{key:<{width}} : {value}")
    print("=" * 35)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    result = benchmark_prompt_ollama(
        args.entrance_url,
        args.model,
        args.prompt,
        request_timeout=args.request_timeout,
        thinking=args.thinking,
        max_attempts=args.max_attempts,
    )

    if args.json:
        print(json.dumps(dataclasses.asdict(result), ensure_ascii=False, indent=2))
    else:
        print_result(result)

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

