# llm_bench — Olares LLM 基准测试脚本

`scripts/llm_bench/llm_bench.py` 是一个**端到端的、可被 cron 触发的 LLM 基准测试脚本**，专门用来批量评估 Olares 上以 Ollama / vLLM / LLaMA.cpp chart 形式部署的大模型。

只依赖 Python 3.9+ 标准库（`urllib`、`subprocess`、`smtplib`、`json` 等），无需 `pip install`，可以直接 cron 起来定时跑。

---

## 它做了什么

对配置文件 `models[]` 里的每一个模型，按下面的流水线**串行**跑一遍（GPU 一次只能装一个，所以必须串行）：

```
┌──────────────────────────────────────────────────────────────────┐
│  1. ensure_installed                                             │
│     ── 检查 chart 当前 state                                     │
│        running   →  reuse（skip_install_if_running=true 时）     │
│        installing/pending →  --watch 等到终态                    │
│        failed/stopped     →  uninstall(--delete-data) + reinstall│
│        not installed      →  market install --watch              │
│  2. find_entrance                                                │
│     ── settings apps get -o json 拿 entrances[].url + authLevel  │
│        没拿到 URL 就用 ports[] + namespace 拼 cluster 内部 URL   │
│        per-model 的 endpoint_url 优先级最高                      │
│  3. ensure_entrance_public                                       │
│     ── authLevel != public 时自动 flip                           │
│        默认开关 auto_open_internal_entrance=true                 │
│  4. (api_type=ollama) pull_model_ollama                          │
│     ── POST /api/pull 触发权重下载（chart 自带权重的可关闭）     │
│  5. warmup                                                       │
│     ── 一次小请求把模型装进显存，下面才好测                      │
│  6. 串行跑 questions[]，每条 prompt 录一条 QuestionResult        │
│     ── ollama: /api/generate stream=false                        │
│     ── openai: /v1/chat/completions stream=false                 │
│  7. uninstall                                                    │
│     ── 默认 uninstall_after_run=true，腾 GPU/磁盘给下个模型      │
│     ── preserve_if_existed=true 时只跳过预装的                   │
└──────────────────────────────────────────────────────────────────┘
```

跑完所有模型后：

- 把所有 `ModelResult` 序列化成 JSON 写到 `output_dir/llm_bench_<timestamp>.json`
- 渲染一张含汇总表 + per-prompt 明细的 HTML 到 `output_dir/llm_bench_<timestamp>.html`
- （可选）通过 SMTP 把 HTML 当邮件正文 + 把 JSON 当附件发给配置里的 `email.to`

---

## 两条 backend 的差异

`api_type` 字段决定走哪条路径：


| 维度      | `ollama`（默认）                                           | `openai`（vLLM / llama.cpp / 其他 OAI-compat）                                                          |
| ------- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| 端点      | `/api/generate`（也支持 `/api/chat`）                       | `/v1/chat/completions`（或 `/v1/completions`）                                                         |
| stream  | `false`，server 一次性返回完整 JSON                            | `false`（强制；改 true 会破坏 JSON 解析）                                                                      |
| TTFT    | 精确：`load_duration + prompt_eval_duration`              | 近似：单独发一个 `max_tokens=1` 的请求测 round-trip                                                             |
| TPS     | 精确：`eval_count / eval_duration`（decode-only，server 报的） | llama.cpp 有 `timings` 块时取 server TPS；否则退化为 client end-to-end TPS                                    |
| token 数 | server 报 `eval_count` / `prompt_eval_count`            | `usage.completion_tokens` 等；缺失时按字符兜底估算                                                              |
| 模型权重    | 默认 `pull_model=true`，调 `/api/pull` 下载                  | chart 自带权重，配置里 `pull_model=false`                                                                   |
| 可调采样参数  | 仅 prompt + stream                                      | `max_tokens` / `temperature` / `top_p` / `extra_body` 等通过 `openai_defaults` + per-model `openai` 块配 |
| Auth 头  | 不发                                                     | `Bearer <api_key>`（默认 `EMPTY` / 空 = 不发，跟 curl 行为一致）                                                 |


OpenAI 路径专门参考了 `scripts/llm_api_benchmark.py` 的实现，确保两个脚本对同一台 vLLM/llama.cpp 跑出来的数能直接比对。

---

## 输出指标

每条 prompt 都是一行 `QuestionResult`：


| 字段                               | ollama             | openai                                   | 含义                             |
| -------------------------------- | ------------------ | ---------------------------------------- | ------------------------------ |
| `wall_seconds`                   | ✓                  | ✓                                        | 客户端端到端 round-trip              |
| `ttft_seconds`                   | server 精确          | max_tokens=1 近似                          | 第一个 token 出现时间                 |
| `eval_count`                     | server 报           | `usage.completion_tokens` 或字符估算          | 生成的 token 数                    |
| `eval_seconds`                   | server 报           | llama.cpp `timings.predicted_ms` 才有      | decode 用时                      |
| `tps`                            | server decode-only | server（llama.cpp）或 client（vLLM）          | 主要 TPS 指标                      |
| `prompt_tokens` / `total_tokens` | 0                  | usage 块                                  | 输入 / 总 token                   |
| `client_tps`                     | 同 tps              | `eval_count / wall_seconds`              | 端到端 TPS（含 prefill+网络）          |
| `server_tps_reported`            | 0                  | llama.cpp `timings.predicted_per_second` | server 自报 TPS                  |
| `tokens_estimated`               | false              | true 时表示 token 数是字符估算的                   | 数据可信度提示                        |
| `note`                           | 短                  | 长                                        | 解释每行哪些指标是近似的，方便 reading report |


每个模型聚合成一个 `ModelResult`，汇总表里取所有成功 prompt 的平均值。

---

## 运行依赖

1. `**olares-cli` 已经登录过**：脚本通过 subprocess 调 `olares-cli`，CLI 内部会从 `~/.olares-cli/config.json` 读 profile 元数据 + 从 OS keychain 读 access/refresh token。第一次跑前必须：
  ```bash
   olares-cli profile login --olares-id <id>
   # 或
   olares-cli profile import --olares-id <id> --refresh-token <tok>
  ```
   cron 场景注意 `HOME` / macOS keychain unlock 这两个坑，详见 `cmd.md`。
2. **HuggingFace token 已配（vLLM 模型用）**：vLLM chart 需要的 user 级 env 在跑脚本前一次性配好，脚本不再帮你配：
  ```bash
   olares-cli settings advanced env user set --var OLARES_USER_HUGGINGFACE_TOKEN=hf_xxx
  ```
3. `**olares-cli` 二进制路径可配**：`--cli-path /home/olares/test/olares-cli` 或 config 里 `"cli_path": "..."`，默认从 PATH 找 `olares-cli`。

---

## 配置文件

完整示例见 `config.example.json`，最小可跑配置见 `1.json`。两类配置项：

### 全局默认（每个 model 可覆盖）


| 字段                            | 默认            | 说明                                                    |
| ----------------------------- | ------------- | ----------------------------------------------------- |
| `cli_path`                    | `olares-cli`  | `olares-cli` 二进制路径                                    |
| `install_timeout_minutes`     | 90            | `market install --watch` 超时                           |
| `uninstall_timeout_minutes`   | 30            | `market uninstall --watch` 超时                         |
| `request_timeout_seconds`     | 1800          | 单次推理 HTTP 请求超时                                        |
| `pull_timeout_seconds`        | 3600          | `/api/pull` + warmup 超时                               |
| `delete_data`                 | true          | uninstall 时是否带 `--delete-data` 释放磁盘                   |
| `pull_model`                  | true          | api_type=ollama 时是否调 `/api/pull`                      |
| `auto_open_internal_entrance` | true          | entrance 不是 public 时是否自动 flip 到 public                |
| `set_public_during_run`       | false         | legacy 别名，true 时等同 `auto_open_internal_entrance=true` |
| `skip_install_if_running`     | true          | 已经 running 的 chart 跳过 install                         |
| `preserve_if_existed`         | false         | true 时跑前已存在的 chart 跑完不卸载                              |
| `uninstall_after_run`         | true          | false 时跑完不卸载（优先级高于 preserve_if_existed）               |
| `cooldown_seconds`            | 30            | 模型之间的休眠                                               |
| `output_dir`                  | config 文件所在目录 | 报告输出目录                                                |
| `openai_defaults`             | (见示例)         | OpenAI-shape backend 的全局采样参数                          |
| `email`                       | required      | SMTP 配置                                               |


### Per-model

最少需要：

```json
{ "app_name": "ollamadeepseekr114bv2", "model_name": "deepseek-r1:14b" }
```

可选：`api_type` (`ollama`|`openai`)、`entrance_name`、`endpoint_url`（手工指定后端 URL，跳过 entrance 自动发现）、`envs`（chart-level `--env KEY=VAL` 列表）、`openai`（覆盖 openai_defaults 的子对象）、上面所有"全局默认"中的字段。

---

## 命令行用法

```bash
# 跑完整流程
python3 llm_bench.py -c config.json

# 跑测试，不发邮件
python3 llm_bench.py -c config.json --no-email

# 仅 dump 每个 chart 的 env 需求（不安装、不卸载）
python3 llm_bench.py -c config.json --probe

# 指定 olares-cli 路径 + 指定日志文件
python3 llm_bench.py -c config.json \
  --cli-path /home/olares/test/olares-cli \
  --log /var/log/llm_bench.log

# cron 示例（每天 3:00 跑一遍）
0 3 * * * cd /home/olares/test && \
  /usr/bin/python3 llm_bench.py -c 1.json \
  --cli-path /home/olares/test/olares-cli \
  --log /var/log/llm_bench.log >>/var/log/llm_bench.cron.log 2>&1
```

---

## 项目布局

```
scripts/llm_bench/
├── llm_bench.py          # 主脚本
├── config.example.json   # 配置示例
├── llm_bench.html        # 测试指标数据html生成
├── llm_bench.json        # 测试指标数据json生成
└── readme.md             # 文档介绍
```

---

## 最终发送邮件效果

![邮件内容mail.png](asset/mail.png)