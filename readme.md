# llm_bench — Olares LLM 基准测试

> v2.0：原 2104 行的单文件 `llm_bench.py` 已按「核心逻辑 + 模型 + 数据 + 工具 + 入口」拆成扁平的 5 桶布局；顶层 `llm_bench.py` 现在 = 入口（argparse + main），不再是壳。

一个**端到端、可被 cron 触发的 LLM 基准测试脚本**，用来批量评估 Olares 上以 Ollama / vLLM / LLaMA.cpp chart 形式部署的大模型。

依赖：
- Python 3.9+
- 官方 [`openai`](https://pypi.org/project/openai/) SDK（≥1.30，用于 OpenAI-compatible 后端；Ollama 后端纯 stdlib）
- `olares-cli` 在 PATH 中（或在配置 / `--cli-path` 里给出绝对路径）

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirement.txt
```

---

## 它做了什么

对配置文件 `models[]` 里的每一个模型，按下面的流水线**串行**跑一遍（GPU 一次只能装一个，所以必须串行）：

```text
1. ensure_installed         core/lifecycle    检查 chart 状态 → reuse / wait / 重装
2. find_entrance            core/entrance     拿 entrance URL，没拿到则用 ports[]+namespace 拼集群内部 URL
3. ensure_entrance_public   core/entrance     authLevel != public 时自动 flip
4. wait_until_api_ready     core/readiness    GET /api/tags 或 /v1/models 等模型真正加载好
5. pull_model_ollama (opt)  core/benchmark    仅 ollama+pull_model:true 时主动调 /api/pull
6. warmup_until_ok          core/readiness    发一次小请求把模型装进显存（带重试）
7. 串行跑 questions[]        core/benchmark    ollama→/api/generate；openai→/v1/chat/completions
8. uninstall                core/lifecycle    默认 uninstall_after_run=true 腾 GPU/磁盘
```

跑完所有模型后：

- `data/report_writer.py` 把所有 `ModelResult` 写到 `output_dir/llm_bench_<timestamp>.json` + `.html`
- `data/mailer.py`（除非加 `--no-email`）通过 SMTP 把 HTML 当邮件正文 + JSON 当附件发给 `email.to`

---

## 项目布局

```text
benchmark_llm_test/
├── llm_bench.py                     # 入口：argparse + main() + 主循环（直接 python3 跑）
├── models.py                        # 模型   QuestionResult / ModelResult / OpenAIConfig
├── core/                            # 核心逻辑
│   ├── lifecycle.py                 #   chart 装/卸 + ensure_installed + 状态桶
│   ├── entrance.py                  #   entrance 发现 + auth 翻转
│   ├── readiness.py                 #   wait_until_api_ready + warmup_until_ok
│   ├── orchestrator.py              #   bench_model() 单模型流水线
│   └── benchmark/
│       ├── ollama.py                #     /api/pull + /api/generate
│       └── openai.py                #     /v1/chat/completions + TTFT 近似 + 配置合并
├── data/                            # 数据
│   ├── config.py                    #   load_config + setup_logging
│   ├── html_report.py               #   render_html
│   ├── report_writer.py             #   写 JSON + HTML
│   ├── mailer.py                    #   SMTP（implicit TLS / STARTTLS 自动判断 + 重试）
│   └── probe.py                     #   --probe：只 dump chart env 需求
├── utils/                           # 工具
│   ├── cli_runner.py                #   olares-cli subprocess 封装
│   ├── http.py                      #   urllib 封装 + OpenAIHTTPError + auth_hint
│   ├── format.py                    #   human_bytes + fmt_duration
│   └── tokens.py                    #   rough_token_count + ms_to_seconds
├── config.example.json              # 简化样例（无注释字段）
├── 2.json                           # 你的工作配置（带 _comment_*，本脚本会忽略）
├── readme.md                        # 本文
├── cmd.md                           # olares-cli 命令速查
└── asset/
```

子模块之间使用相对裸导入（`from utils.cli_runner import cli` / `from core.orchestrator import bench_model`），跑脚本时 Python 自动把 `llm_bench.py` 所在目录加到 `sys.path`，所以不需要再有 `__init__.py` 顶层包标记。

---

## 两条 backend 的差异

`api_type` 字段决定走哪条路径：

| 维度 | `ollama`（默认） | `openai`（vLLM / llama.cpp / 其他 OAI-compat） |
|---|---|---|
| 端点 | `/api/generate`（也支持 `/api/chat`） | `/v1/chat/completions`（或 `/v1/completions`） |
| stream | `false`，server 一次性返回完整 JSON | `false`（强制；改 true 会破坏 JSON 解析） |
| TTFT | 精确：`load_duration + prompt_eval_duration` | 近似：单独发一个 `max_tokens=1` 的请求测 round-trip |
| TPS | 精确：`eval_count / eval_duration`（decode-only） | llama.cpp 有 `timings` 块时取 server TPS；否则退化为 client end-to-end TPS |
| token 数 | server 报 `eval_count` / `prompt_eval_count` | `usage.completion_tokens` 等；缺失时按字符兜底估算 |
| 模型权重 | chart launcher 自动 pull；可设 `pull_model:true` 主动调 `/api/pull` 看进度 | chart 自带权重，配置里 `pull_model:false` |
| 可调采样参数 | 仅 prompt + stream | `max_tokens` / `temperature` / `top_p` / `extra_body` 等通过 `openai_defaults` + per-model `openai` 块配 |
| Auth 头 | 不发 | `Bearer <api_key>`（默认 `EMPTY` / 空 = 不发，跟 curl 行为一致） |

---

## 输出指标（每条 prompt 一行 `QuestionResult`）

| 字段 | ollama | openai | 含义 |
|---|---|---|---|
| `wall_seconds` | ✓ | ✓ | 客户端端到端 round-trip |
| `ttft_seconds` | server 精确 | max_tokens=1 近似 | 第一个 token 出现时间 |
| `eval_count` | server 报 | `usage.completion_tokens` 或字符估算 | 生成的 token 数 |
| `eval_seconds` | server 报 | llama.cpp `timings.predicted_ms` 才有 | decode 用时 |
| `tps` | server decode-only | server（llama.cpp）或 client（vLLM） | 主要 TPS 指标 |
| `prompt_tokens` / `total_tokens` | 0 | usage 块 | 输入 / 总 token |
| `client_tps` | 同 tps | `eval_count / wall_seconds` | 端到端 TPS（含 prefill+网络） |
| `server_tps_reported` | 0 | llama.cpp `timings.predicted_per_second` | server 自报 TPS |
| `tokens_estimated` | false | true 时表示 token 数是字符估算 | 数据可信度提示 |
| `note` | 短 | 长 | 说明本行哪些指标是近似 |

每个模型聚合成一个 `ModelResult`，HTML 报表里取所有成功 prompt 的平均值。

---

## 运行依赖

1. **Python 环境**：
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirement.txt   # 仅一个包：openai>=1.30
   ```
2. **olares-cli 已经登录过**：脚本通过 subprocess 调 `olares-cli`，CLI 内部会从 `~/.olares-cli/config.json` 读 profile 元数据 + 从 OS keychain 读 access/refresh token：
   ```bash
   olares-cli profile login --olares-id <id>
   # 或非交互式：
   olares-cli profile import --olares-id <id> --refresh-token <tok>
   ```
   cron 场景注意 `HOME` / macOS keychain unlock 这两个坑，详见 `cmd.md`。
3. **HuggingFace token 已配（vLLM 系列必用）**：
   ```bash
   olares-cli settings advanced env user set --var OLARES_USER_HUGGINGFACE_TOKEN=hf_xxx
   ```
4. **olares-cli 二进制路径可配**：`--cli-path /home/olares/test/olares-cli` 或 config 里 `"cli_path": "..."`，默认从 PATH 找 `olares-cli`。

---

## 配置文件参考

`config.example.json` 已经精简为可直接复制即跑的最小可用版；下面是**所有**支持字段的完整说明（按出现顺序），从原 `2.json` 的注释整理而来。

### 全局默认（每个 model 可同名字段覆盖）

| 字段 | 默认 | 说明 |
|---|---|---|
| `cli_path` | `olares-cli` | olares-cli 二进制路径。当 cli 装在非常规路径（如 `/home/olares/test/olares-cli`）时在这里指定，或运行时用 `--cli-path` 覆盖 |
| `install_timeout_minutes` | 90 | `market install --watch` 超时（分钟） |
| `uninstall_timeout_minutes` | 30 | `market uninstall --watch` 超时（分钟） |
| `request_timeout_seconds` | 1800 | 单次推理 HTTP 请求超时（秒） |
| `pull_timeout_seconds` | 3600 | `/api/pull` + warmup 的请求超时（秒） |
| `api_ready_timeout_minutes` | 60 | chart 进入 running 后，轮询后端 `/api/tags`（ollama）或 `/v1/models`（openai）直到目标模型上线，最长等多少分钟。这是 `wait_until_api_ready` 等价于 chart launcher UI 上"Waiting for Ollama → Ready to chat"的判据 |
| `api_ready_probe_interval_seconds` | 30 | 上面的轮询每次失败 sleep 多少秒 |
| `warmup_retries` | 10 | warmup（小推理请求）最多重试几次。覆盖 KV cache init / lazy CUDA compile / Modelfile 重量化 / vLLM scheduler warmup 等懒初始化 |
| `warmup_retry_sleep_seconds` | 30 | warmup 每次失败 sleep 多少秒。默认 10×30s ≈ 5 分钟 |
| `pull_max_attempts` | 5 | `/api/pull` transport 失败最多重试几次（仅 `pull_model:true` 时用）。Ollama 服务端的拉取是幂等可断点续传，所以重连等于继续；`fatal`（manifest unknown / OOM-on-disk）不会重试 |
| `pull_retry_sleep_seconds` | 30 | `/api/pull` 重试间隔（秒） |
| `delete_data` | `true` | uninstall 时是否带 `--delete-data` 释放磁盘 |
| `pull_model` | `false` | api_type=ollama 时是否主动调 `/api/pull`。olares-market 上所有 `ollama*` chart 都自带 launcher 容器在启动时跑 `ollama pull <model>`，外面再发一遍是冗余的（双发的 HTTP keepalive 还容易撞 ingress idle timeout）。只有当你测的是裸 ollama-server（没有 chart launcher），或者想把下载进度日志打到本脚本的 stdout 时才开（per-model 也可覆盖） |
| `auto_open_internal_entrance` | `true` | entrance 不是 public 时自动调 `auth-level set --level public` + `policy set --default-policy public`。设置 per-app，模型 uninstall 时跟着销毁，所以不会泄漏 |
| `set_public_during_run` | `false` | legacy 别名，true 时等同 `auto_open_internal_entrance:true` |
| `skip_install_if_running` | `true` | 已 running 的 chart 跳过 install |
| `preserve_if_existed` | `false` | true 时仅"跑前已存在"的 chart 跑完不卸载（优先级低于 `uninstall_after_run`） |
| `uninstall_after_run` | `true` | true（默认）：跑完一个模型立刻 uninstall，腾 GPU/磁盘给下一个；false：跑完一律保留（即使本次新装的）。下次 `skip_install_if_running` 命中可直接复用。优先级高于 `preserve_if_existed` |
| `cooldown_seconds` | 30 | 模型之间的休眠（秒），让 GPU 显存彻底回收 |
| `output_dir` | config 文件所在目录 | JSON+HTML 报告输出目录，自动创建 |
| `openai_defaults` | 见下表 | 全局 openai-shape 采样参数 |
| `email` | **必填** | SMTP 配置，见下表 |

### `openai_defaults` 子对象（仅对 `api_type:openai` 生效）

vLLM / LLaMA.cpp / 其他 OpenAI-compatible 后端共用的默认参数。每个 model 可以在 `openai`: {...} 块里 per-model 覆盖。脚本内部参考 `scripts/llm_api_benchmark.py` 的实现。

| 字段 | 默认 | 说明 |
|---|---|---|
| `api_key` | `EMPTY` | llama-server 启了 `--api-key` 时填进来；vLLM/chart 内部不需要可保持 `EMPTY`（脚本会跳过 `Authorization` 头，跟 curl 行为一致） |
| `endpoint` | `chat` | `chat` 走 `/v1/chat/completions`；`completion` 走 `/v1/completions` |
| `max_tokens` | 256 | 推理生成上限 |
| `temperature` | 0.0 | 采样温度 |
| `top_p` | `null` | nucleus 采样；`null` 表示不发 |
| `extra_headers` | `{}` | 整体合并进请求头 |
| `extra_body` | `{}` | 整体合并进 payload，例如 `{"top_k":50,"repetition_penalty":1.05}` |
| `measure_ttft_approx` | `true` | true 时在每条 prompt 前先发一个 `max_tokens=1` 请求测 round-trip 近似 TTFT（stream=false 拿不到真 TTFT） |

### `email` 子对象（必填）

Gmail 必须用 App Password（在 Google Account → Security → 2-Step Verification → App passwords 里生成）。配置文件含密码，建议 `chmod 600`。

| 字段 | 默认 | 说明 |
|---|---|---|
| `smtp_host` | — | SMTP 主机名 |
| `smtp_port` | — | 465（implicit TLS）/ 587（STARTTLS） |
| `use_ssl` | `port==465` | 显式覆盖；不写时用启发式：465→implicit TLS，其他→STARTTLS |
| `smtp_timeout` | 120 | TCP / TLS 握手超时（秒） |
| `smtp_retries` | 3 | 仅对 transport 错误重试；auth/protocol 错立即抛 |
| `smtp_retry_backoff` | 5 | 退避基数（秒），按 `min(backoff*attempt, 60)` 增长 |
| `username` / `password` | — | SMTP 凭据 |
| `from` / `to` | — | `to` 支持逗号分隔多个收件人 |
| `subject` | `Olares LLM benchmark <date>` | 邮件主题 |

### Per-model（最少 2 个字段）

```json
{ "app_name": "ollamadeepseekr114bv2", "model_name": "deepseek-r1:14b" }
```

可选字段：

- `api_type`：`ollama` | `openai`
- `entrance_name`：指定 entrance 名（多 entrance 时挑一个）
- `endpoint_url`：直接覆盖端点 URL，**跳过 entrance 自动发现 + 跳过 auth-level 翻转**。例如 `"http://localhost:30888"`（NodePort）或 `"http://10.x.y.z:8000"`（Pod IP）
- `envs`：数组，原样传给 `olares-cli market install --env KEY=VALUE`
- `openai`：子对象，仅 `api_type:openai` 生效，覆盖 `openai_defaults` 里的 `api_key/endpoint/max_tokens/temperature/top_p/extra_headers/extra_body/measure_ttft_approx`
- 上表「全局默认」中的所有字段（`install_timeout_minutes` / `delete_data` / `pull_model` / ...）

> ⚠️ `model_name` 必须和后端实际加载的模型名**一字不差**。第一次跑前用 `python3 llm_bench.py -c <conf> --probe` 看 chart 声明，或装好后用 `ollama list`(ollama) / `curl URL/v1/models`(vllm 或 llamacpp) 取准确 ID 回填。

### vLLM / llama.cpp 内部 entrance

vLLM / llama.cpp 这类 chart 默认 `authLevel=internal`，没有公网域名。脚本会自动用 `olares-cli settings apps get <app>` 拿到 `ports[] + namespace` 拼出 cluster 内部 URL（`http://<svc>.<ns>:<port>`），从 Olares host 上一般能直接打通。如果你的环境从 host 解不到 cluster DNS，可以：

1. 在该 model 上设置 `endpoint_url` 手动指定 URL；或
2. 打开 `auto_open_internal_entrance:true`（默认）让脚本临时把 entrance 改成 public。

### 一些常见 chart 备忘（来自原 `2.json` 注释）

- BGE-M3、bge-* 系列是 embedding 模型，不支持 `/api/generate`；脚本会得到 4xx 报错并标 ERR。如要 benchmark embedding 性能需要单独走 `/api/embed`，目前脚本不覆盖。
- LLaVA 1.6、MiniCPM-V 是多模态视觉模型，本脚本只发文本 prompt 不发图片，能跑但只是普通 chat 路径。
- Gemma4 / Qwen3.5 等非官方量化（Unsloth Dynamic Q4 等）`model_name` 多半是 chart Modelfile 里 `hf.co/<repo>` 形式，强烈建议先用 `--probe` 看 `GEMMA_MODEL` / `MODEL_NAME` 这类 env 默认值。
- vLLM 系列需要先在 OS 配 HF token（见上面"运行依赖"第 2 条）。
- llama.cpp 120B 模型超大，建议 `install_timeout_minutes` ≥ 240。

---

## 命令行用法

```bash
# 跑完整流程
python3 llm_bench.py -c config.example.json

# 跑测试，不发邮件
python3 llm_bench.py -c config.example.json --no-email

# 仅 dump 每个 chart 的 env 需求（不安装、不卸载）
python3 llm_bench.py -c config.example.json --probe

# 指定 olares-cli 路径 + 指定日志文件
python3 llm_bench.py -c config.example.json \
  --cli-path /home/olares/test/olares-cli \
  --log /var/log/llm_bench.log

# cron 示例（每天 3:00 跑一遍）
0 3 * * * cd /home/olares/test && \
  /usr/bin/python3 llm_bench.py -c 2.json \
  --cli-path /home/olares/test/olares-cli \
  --log /var/log/llm_bench.log >>/var/log/llm_bench.cron.log 2>&1
```

---

## 嵌入到其他脚本

把项目根目录加到 `sys.path` 之后即可按桶导入：

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path("/path/to/benchmark_llm_test")))

from data.config import load_config
from core.orchestrator import bench_model
from models import ModelResult

cfg = load_config("config.example.json")
results: list[ModelResult] = [
    bench_model(spec, cfg["questions"], cfg) for spec in cfg["models"]
]
```

---

## 最终发送邮件效果

![邮件内容mail.png](asset/mail.png)
