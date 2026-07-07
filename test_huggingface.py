#!/usr/bin/env python3
"""
llm_perf_tester.py
==================
通过 OpenAI 兼容的流式 API 测量本地 llama.cpp 服务的性能指标：

  - Thinking-TTFT   (Time to First Token)      : 从发送请求到收到第一个非空 token 的耗时
                                                 （thinking 或 content，取最早到达的那个）
  - TTFT            (Time to First Content)    : 从发送请求到收到第一个非空 content token 的耗时
                                                 （跨越完整的 thinking 阶段）
  - TPS             (Tokens Per Second)        : 全部 token（thinking + content）的生成吞吐量
                                                 计算方式：total_tokens / (t_last - t_first)
  - TPOT            (Time Per Output Token)    : 平均每个输出 token 的生成时间（毫秒/token）
                                                 计算方式：(t_last - t_first) / total_tokens，即 TPS 的倒数

使用方法：
  python3 llm_perf_tester.py [选项]

选项：
  --url         API 端点，默认 http://127.0.0.1:8080/v1/chat/completions
  --model       模型名称，默认 default
  --prompt      测试 prompt，默认为一个简短的问题
  --max-tokens  最大生成 token 数，默认 512
  --api-key     API Key，llama.cpp 本地服务通常不需要，默认 sk-no-key
  --runs        重复测量次数并输出均值，默认 1
"""

import requests
import time
import json
import argparse
import sys
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class PerfResult:
    thinking_ttft: float = 0.0      # 收到第一个任意非空 token 的延迟（秒），即传统意义上的首 token 延迟
    ttft: float = 0.0               # 收到第一个 content 非空 token 的延迟（秒），跨越完整 thinking 阶段
    tps: float = 0.0                # 全部 token 的生成吞吐量（tokens/s）
    tpot: float = 0.0               # 平均每个 token 的生成时间（秒/token），即 TPS 的倒数
    total_time: float = 0.0         # 总耗时（秒）
    thinking_tokens: int = 0        # thinking 阶段 token 数
    content_tokens: int = 0         # 内容阶段 token 数
    total_tokens: int = 0           # 总 token 数
    has_thinking: bool = False      # 是否检测到 thinking 内容


# ─────────────────────────────────────────────
# 核心测量函数
# ─────────────────────────────────────────────

def measure_once(url: str, model: str, prompt: str,
                 api_key: str = "sk-no-key",
                 max_tokens: int = 512) -> PerfResult:
    """
    发送一次流式请求并采集性能指标。

    llama.cpp 的流式响应遵循 OpenAI SSE 格式，每行形如：
        data: {"id":"...","choices":[{"delta":{"content":"..."},...}],...}
    带 thinking 能力的模型（如 DeepSeek-R1、Qwen3）会额外输出：
        delta.reasoning_content  —— thinking 阶段的 token
        delta.content            —— 正式回答阶段的 token
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
    }

    result = PerfResult()

    # ── 时间戳记录 ──────────────────────────────
    t_request_sent: float = 0.0
    t_first_any: Optional[float] = None        # 第一个任意非空 token（thinking 或 content）→ Thinking-TTFT
    t_first_content: Optional[float] = None    # 第一个 content 非空 token → TTFT
    t_last_token: Optional[float] = None       # 最后一个 token

    try:
        t_request_sent = time.perf_counter()
        resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] 无法连接到 {url}，请确认 llama.cpp server 已启动。")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] HTTP 错误: {e}")
        # 打印服务端返回的具体错误内容，便于定位 400/422 等问题
        try:
            body = resp.text
            if body:
                print(f"[ERROR] 服务端响应内容: {body}")
        except Exception:
            pass
        sys.exit(1)

    # ── 逐行解析 SSE 流 ─────────────────────────
    try:
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue

            line = raw_line.decode("utf-8", errors="replace")

            if not line.startswith("data: "):
                continue

            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices")
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            now = time.perf_counter()

            # ── Thinking token（reasoning_content 字段）──
            reasoning = delta.get("reasoning_content") or ""
            if reasoning:
                result.has_thinking = True
                result.thinking_tokens += 1
                result.total_tokens += 1
                t_last_token = now
                if t_first_any is None:
                    t_first_any = now   # Thinking-TTFT：第一个任意 token

            # ── Content token（content 字段）─────────────
            content = delta.get("content") or ""
            if content:
                result.content_tokens += 1
                result.total_tokens += 1
                t_last_token = now
                if t_first_any is None:
                    t_first_any = now   # 无 thinking 时，content 也算第一个 token
                if t_first_content is None:
                    t_first_content = now  # TTFT：第一个 content token

    except Exception as e:
        print(f"[WARN] 流读取异常: {e}")

    t_end = time.perf_counter()

    # ── 计算各项指标 ────────────────────────────

    # Thinking-TTFT：从请求发出到第一个任意非空 token（thinking 或 content）
    if t_first_any is not None:
        result.thinking_ttft = t_first_any - t_request_sent

    # TTFT：从请求发出到第一个 content 非空 token（跨越完整 thinking 阶段）
    if t_first_content is not None:
        result.ttft = t_first_content - t_request_sent
    elif t_first_any is not None:
        # 无 thinking 的普通模型，TTFT == Thinking-TTFT
        result.ttft = result.thinking_ttft

    # TPS：total_tokens / (t_last - t_first_any)
    # 包含 thinking + content 全部 token，排除 prefill 阶段
    if t_first_any is not None and t_last_token is not None:
        decode_duration = t_last_token - t_first_any
        if decode_duration > 0 and result.total_tokens > 0:
            result.tps = result.total_tokens / decode_duration
            result.tpot = decode_duration / result.total_tokens
        else:
            result.tps = 0.0
            result.tpot = 0.0

    result.total_time = t_end - t_request_sent
    return result


# ─────────────────────────────────────────────
# 输出格式化
# ─────────────────────────────────────────────

def print_result(result: PerfResult, run_index: Optional[int] = None):
    header = "Performance Metrics"
    if run_index is not None:
        header = f"Run #{run_index + 1} — {header}"

    print()
    print("=" * 50)
    print(f"  {header}")
    print("=" * 50)
    print(f"  Total time       : {result.total_time:.3f} s")
    print(f"  Total tokens     : {result.total_tokens}")
    if result.has_thinking:
        print(f"  Thinking tokens  : {result.thinking_tokens}")
        print(f"  Content tokens   : {result.content_tokens}")
    print("-" * 50)
    print(f"  Thinking-TTFT    : {result.thinking_ttft * 1000:.1f} ms  (first any token)")
    if result.has_thinking:
        print(f"  TTFT             : {result.ttft * 1000:.1f} ms  (first content token)")
        thinking_dur = result.ttft - result.thinking_ttft
        print(f"  Thinking duration: {thinking_dur * 1000:.1f} ms")
    else:
        print(f"  TTFT             : {result.ttft * 1000:.1f} ms  (= Thinking-TTFT, no thinking phase)")
    print(f"  TPS              : {result.tps:.2f} tokens/s  (all tokens / decode time)")
    print(f"  TPOT             : {result.tpot * 1000:.2f} ms/token  (avg time per output token)")
    print("=" * 50)


def print_summary(results: list):
    if len(results) <= 1:
        return

    n = len(results)
    avg_thinking_ttft = sum(r.thinking_ttft for r in results) / n
    avg_ttft = sum(r.ttft for r in results) / n
    avg_tps = sum(r.tps for r in results) / n
    avg_tpot = sum(r.tpot for r in results) / n
    avg_total = sum(r.total_time for r in results) / n

    print()
    print("=" * 50)
    print(f"  Summary ({n} runs)")
    print("=" * 50)
    print(f"  Avg Total time   : {avg_total:.3f} s")
    print(f"  Avg Thinking-TTFT: {avg_thinking_ttft * 1000:.1f} ms")
    if any(r.has_thinking for r in results):
        print(f"  Avg TTFT         : {avg_ttft * 1000:.1f} ms")
    print(f"  Avg TPS          : {avg_tps:.2f} tokens/s")
    print(f"  Avg TPOT         : {avg_tpot * 1000:.2f} ms/token")
    print("=" * 50)


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="测量本地 llama.cpp 服务的 TTFT / Thinking-TTFT / TPS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8080/v1/chat/completions",
        help="API 端点（默认: http://127.0.0.1:8080/v1/chat/completions）",
    )
    parser.add_argument(
        "--model",
        default="default",
        help="模型名称（默认: default）",
    )
    parser.add_argument(
        "--prompt",
        default="请用简洁的语言解释什么是量子计算。",
        help="测试 prompt（默认: 请用简洁的语言解释什么是量子计算。）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        dest="max_tokens",
        help="最大生成 token 数（默认: 512）",
    )
    parser.add_argument(
        "--api-key",
        default="sk-no-key",
        dest="api_key",
        help="API Key（本地 llama.cpp 通常不需要，默认: sk-no-key）",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="重复测量次数（默认: 1）",
    )

    args = parser.parse_args()

    print(f"Target  : {args.url}")
    print(f"Model   : {args.model}")
    print(f"Prompt  : {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print(f"Runs    : {args.runs}")

    results = []
    for i in range(args.runs):
        if args.runs > 1:
            print(f"\n[Run {i + 1}/{args.runs}] Sending request...")
        else:
            print("\nSending request...")

        result = measure_once(
            url=args.url,
            model=args.model,
            prompt=args.prompt,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
        )
        results.append(result)
        print_result(result, run_index=i if args.runs > 1 else None)

    print_summary(results)


if __name__ == "__main__":
    main()
