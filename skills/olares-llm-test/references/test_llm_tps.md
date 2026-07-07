# 功能一：测试 entrance-url 对应模型的 TPS

测量目标模型的性能指标（TPS、TTFT、TPOT 等）。先完成 `SKILL.md` 中的「准备环境」与「前置检查」（应用 running 且已装载、认证 public），并已 `source venv/bin/activate`。

根据应用类型选择脚本：

- **ollama 类型** → `test_ollama.py`：只需替换 `--entrance-url` 和 `--model`。
- **其他类型（llama.cpp / vLLM 等 OpenAI 兼容）** → `test_huggingface.py`：把 `/v1` 之前的 URL 替换为 entrance-url，并替换 `--model`。

## ollama 类型

```bash
python test_ollama.py --entrance-url https://74bfa5ee.yaotest004.olares.com --model gemma4:26b --prompt "请从概念、发展和应用方面展开说明什么是人工智能。"
```

> 替换要点：`--entrance-url` 换成当前应用的 entrance-url，`--model` 换成对应模型名称，`--prompt` 可保持不变。

预期结果（示例）：

```
===== QuestionResult 全部指标 =====
prompt                : 请从概念、发展和应用方面展开说明什么是人工智能。
ok                    : True
error                 : None
response_chars        : 2482
server_model          : gemma4:26b
wall_seconds          : 18.12
ttft_seconds          : 8.207
thinking_ttft_seconds : 0.0
has_thinking          : False
load_seconds          : 0.228
prompt_eval_seconds   : 2.714
eval_count            : 2017
eval_seconds          : 14.793
tps                   : 136.35
total_server_seconds  : 17.756
prompt_tokens         : 0
total_tokens          : 0
client_tps            : 111.32
server_tps_reported   : 0.0
tokens_estimated      : False
note                  :
===================================
```

判定：`ok : True` 即调用成功；核心性能指标看 `tps`（服务端吞吐）与 `client_tps`（客户端吞吐）、`ttft_seconds`（首 token 延迟）。

## 其他类型（OpenAI 兼容）

```bash
python3 test_huggingface.py \
 --url https://f042fe0a.yaotest004.olares.com/v1/chat/completions \
 --model ornith-1.0-35b \
 --prompt "请从概念、发展和应用方面展开说明什么是人工智能。"
```

> 替换要点：把 `--url` 中 `/v1` 之前的部分（`https://f042fe0a.yaotest004.olares.com`）替换为当前应用的 entrance-url，保留 `/v1/chat/completions` 路径；`--model` 换成对应模型名称。

预期结果（示例）：

```
Target  : https://f042fe0a.yaotest004.olares.com/v1/chat/completions
Model   : ornith-1.0-35b
Prompt  : 请从概念、发展和应用方面展开说明什么是人工智能。
Runs    : 1

Sending request...

==================================================
  Performance Metrics
==================================================
  Total time       : 16.846 s
  Total tokens     : 2798
  Thinking tokens  : 1494
  Content tokens   : 1304
--------------------------------------------------
  Thinking-TTFT    : 939.2 ms  (first any token)
  TTFT             : 9455.6 ms  (first content token)
  Thinking duration: 8516.4 ms
  TPS              : 175.90 tokens/s  (all tokens / decode time)
  TPOT             : 5.69 ms/token  (avg time per output token)
==================================================
```

判定：成功打印 `Performance Metrics` 即调用成功；核心指标看 `TPS`（全部 token / 解码时间）、`TTFT`（首个内容 token 延迟）、`TPOT`（平均每 token 生成时间）。
