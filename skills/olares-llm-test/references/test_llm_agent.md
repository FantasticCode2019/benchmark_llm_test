# 功能二：测试 entrance-url 对应模型的 Agent 能力

测试模型是否具备 tool 选择与调用（tool / function calling）能力。先完成 `SKILL.md` 中的「准备环境」与「前置检查」（应用 running 且已装载、认证 public），并已 `source venv/bin/activate`。

## 第一步：修改 agent_config.json

编辑 `agent_config.json`，把 `base_url` 改成当前应用的 entrance-url（带 `/v1`），把 `model` 改成对应模型名称：

```json
{
  "base_url": "https://30cd0ba8.yuetest002.olares.com/v1",
  "api_key": "empty",
  "model": "Qwen/Qwen3.5-2B",
  "temperature": 0.2,
  "max_tool_rounds": 5,
  "timeout_seconds": 60
}
```

> 替换要点：`base_url` 换成当前应用的 entrance-url（保留末尾 `/v1`），`model` 换成对应模型名称；public 应用 `api_key` 保持 `empty` 即可。

## 第二步：运行测试

```bash
python3 agent.py   --mode tools   --question "我明天去杭州，天气怎么样？请再推荐 3 个适合初次游玩的景点。"
```

预期结果（示例）：

```
{'id': '88eab52d11a4424d934f2f634f0691fc', 'object': 'chat.completion', 'created': 1781092441, 'model': 'Qwen/Qwen3.5-2B', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': None, 'reasoning_content': '<tool_call>\n<function=query_city_weather>\n<parameter=city>\n杭州\n</parameter>\n<parameter=date>\n明天\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=query_city_attractions>\n<parameter=city>\n杭州\n</parameter>\n<parameter=preference>\n亲子\n</parameter>\n<parameter=top_k>\n3\n</parameter>\n</function>\n</tool_call>', 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 248046}], 'usage': {'prompt_tokens': 593, 'total_tokens': 684, 'completion_tokens': 91, 'prompt_tokens_details': None, 'reasoning_tokens': 91}, 'metadata': {'weight_version': 'default'}}
[tools] 第 1 轮：模型通过非标准 reasoning_content/content请求调用 2 个工具。
[tools] 调用工具：query_city_weather({"city": "杭州", "date": "明天"})
[tools] 工具结果：{"ok": true, "result": {"city": "杭州", "date": "明天", "condition": "小雨", "temperature": 24, "temperature_unit": "°C", "humidity": "82%", "wind": "西南风 2 级", "source": "mock_weather_db"}}
[tools] 调用工具：query_city_attractions({"city": "杭州", "preference": "亲子", "top_k": 3})
[tools] 工具结果：{"ok": true, "result": {"city": "杭州", "preference": "亲子", "attractions": ["西湖", "灵隐寺", "西溪国家湿地公园"], "tips": "以上为 mock 数据，适合用于验证模型是否会主动调用工具。", "source": "mock_attractions_db"}}
{'id': '37d67610486744d68bec74c61fb7a5c3', 'object': 'chat.completion', 'created': 1781092443, 'model': 'Qwen/Qwen3.5-2B', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': None, 'reasoning_content': '明天去杭州，天气状况如下：\n*   天气：小雨\n*   温度：24°C\n*   湿度：82%\n*   风力：西南风 2 级\n\n适合初次游玩的亲子景点推荐：\n1.  西湖：杭州的标志性景点，湖光山色优美，适合拍照和休闲。\n2.  灵隐寺：位于西湖之西，拥有著名的飞来峰，文化底蕴深厚，环境清幽。\n3.  西溪国家湿地公园：拥有独特的湿地生态，适合散步和观察自然，环境非常宁静。\n\n祝您旅途愉快！', 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 248046}], 'usage': {'prompt_tokens': 841, 'total_tokens': 981, 'completion_tokens': 140, 'prompt_tokens_details': None, 'reasoning_tokens': 140}, 'metadata': {'weight_version': 'default'}}
[tools] 第 2 轮：模型未继续调用工具，输出最终答案。

===== FINAL ANSWER =====
明天去杭州，天气状况如下：
天气：小雨
温度：24°C
湿度：82%
风力：西南风 2 级

适合初次游玩的亲子景点推荐：
西湖：杭州的标志性景点，湖光山色优美，适合拍照和休闲。
灵隐寺：位于西湖之西，拥有著名的飞来峰，文化底蕴深厚，环境清幽。
西溪国家湿地公园：拥有独特的湿地生态，适合散步和观察自然，环境非常宁静。

祝您旅途愉快！
```

## 判定

- 日志中出现 `[tools] ... 请求调用 N 个工具` 以及 `调用工具：query_city_weather(...)` / `query_city_attractions(...)`，说明模型能主动选择并调用工具。
- 部分 OpenAI 兼容服务不会填充标准 `tool_calls` 字段，而是把 `<tool_call>...</tool_call>` 放在 `reasoning_content` / `content` 中，日志会提示「通过非标准 reasoning_content/content请求调用」，脚本会自动兼容解析，同样视为具备 Agent 能力。
- 最终在 `===== FINAL ANSWER =====` 下基于工具返回结果给出答案，即完整走通一轮 tool calling。
- 若始终未出现工具调用、直接给出答案，或报错缺少 `choices` 字段，则说明该模型 / 服务不支持（或未正确启用）tool calling。

> `--mode functions` 可测试旧版 `functions` / `function_call` 协议，用法同上。
