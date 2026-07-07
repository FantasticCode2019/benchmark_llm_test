---
name: olares-llm-test
description: "测试 Olares 上已部署 LLM 应用的两项能力 — 功能一：测量指定 entrance-url 模型的 TPS / TTFT 等性能指标（ollama 类型用 test_ollama.py，其他类型用 test_huggingface.py）；功能二：用 agent.py 测试模型是否具备 tool / function calling 的 Agent 能力。基于 benchmark_llm_test 仓库运行。Use for LLM 性能测试, 模型 TPS 测试, tool calling / agent 能力测试, olares 模型测试, benchmark llm."
compatibility: 需要 git、python3 (>=3.9) 与网络访问；目标 LLM 应用需处于 running 且已装载模型，认证级别为 public。
---

# olares-llm-test

对 Olares 上已部署并对外提供 OpenAI / Ollama 协议的 LLM 应用做两类测试：

1. **性能（TPS）测试** — 见 [references/test_llm_tps.md](references/test_llm_tps.md)
2. **Agent（tool / function calling）能力测试** — 见 [references/test_llm_agent.md](references/test_llm_agent.md)

两个功能点都必须在下面准备好的 Python 虚拟环境（venv）中运行。

## When to use

- 测试某个 entrance-url 对应模型的 TPS / TTFT / TPOT 等性能指标
- 测试某个 entrance-url 对应模型是否具备 tool / function calling（Agent）能力
- 对 Olares 上的 ollama、llama.cpp、vLLM 等 OpenAI 兼容 LLM 应用做基准测试

## 准备环境（两个功能点的共同前置）

先下载仓库并在其中建立虚拟环境、安装依赖，后续所有命令都在该环境中执行。

```bash
git clone https://github.com/FantasticCode2019/benchmark_llm_test
cd benchmark_llm_test
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **若本地已下载 `benchmark_llm_test` 仓库，则无需再次下载**，直接 `cd` 进入该目录，跳过 `git clone` 这一步。
> 每开一个新终端都需要重新 `source venv/bin/activate`。

## 前置检查（运行任一功能点之前必须确认）

1. **应用已正常启动且已装载模型** —— 目标 LLM 应用在 Olares 上处于 `running` 状态，且模型权重已加载完成（可对外响应请求），否则测试会连接失败或超时。
2. **认证级别为 public** —— 确认该应用的认证级别（entrance auth level）为公开 `public`。只有 public 级别才能在不携带 Olares 登录凭证的情况下直接用 entrance-url 访问；否则请求会被网关拦截（302/401）。

只有以上两点都满足，才继续执行下面的功能点。

## 功能点选择

| 应用类型 | 功能一（TPS） | 说明 |
|---|---|---|
| ollama 类型 | `test_ollama.py` | 只需替换 `--entrance-url` 和 `--model` |
| 其他类型（llama.cpp / vLLM 等 OpenAI 兼容） | `test_huggingface.py` | 替换 `/v1` 之前的 URL 为 entrance-url，并替换 `--model` |

- **功能一：TPS 测试** —— 按应用类型选择脚本，详细命令与结果解读见 [references/test_llm_tps.md](references/test_llm_tps.md)。
- **功能二：Agent 能力测试** —— 先改好 `agent_config.json`（把 `base_url` 改成当前应用的 entrance-url、把 `model` 改成对应模型名），再运行 `agent.py`，详见 [references/test_llm_agent.md](references/test_llm_agent.md)。

## 结果判定

- **TPS 测试**：`ok=True`（ollama）或成功打印 Performance Metrics（其他类型）即视为可用；重点关注 `tps` / `TPS`、`ttft_seconds` / `TTFT`。
- **Agent 测试**：若日志中出现 `调用工具：query_city_weather(...)` / `query_city_attractions(...)` 且最终给出基于工具结果的答案，说明模型具备 tool 选择与调用能力（含通过非标准 `reasoning_content` 触发的兼容情况）。
