#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weather_travel_agent.py

一个最小但完整的 OpenAI 协议 Agent 示例，用于测试模型是否具备：
1. tool calling 能力，也就是 OpenAI Chat Completions API 中的 tools / tool_calls；
2. function calling 兼容能力，也就是旧版 functions / function_call。

内置两个 mock 工具：
- query_city_weather：查询某个城市的天气；
- query_city_attractions：查询某个城市的旅游景点。

运行示例：
    python3 agent.py --question "我明天去杭州，天气怎么样？顺便推荐几个景点" --mode tools
    python3 agent.py --question "查询北京天气和景点" --mode functions

配置方式：
    优先读取 config.json；敏感信息建议通过环境变量提供：
    export OPENAI_API_KEY="你的 API Key"
    export OPENAI_API_BASE="https://api.openai.com/v1"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import requests

DEFAULT_CONFIG = {
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini",
    "temperature": 0.2,
    "max_tool_rounds": 5,
    "timeout_seconds": 60,
}

@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.2
    max_tool_rounds: int = 5
    timeout_seconds: int = 60

def load_config(config_path: str) -> LLMConfig:
    """从配置文件和环境变量加载配置。环境变量优先级高于配置文件。"""
    file_config: Dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            file_config = json.load(f)

    merged = {**DEFAULT_CONFIG, **file_config}

    api_key = os.getenv("OPENAI_API_KEY") or merged.get("api_key") or ""
    base_url = os.getenv("OPENAI_API_BASE") or merged.get("base_url") or DEFAULT_CONFIG["base_url"]
    model = os.getenv("OPENAI_MODEL") or merged.get("model") or DEFAULT_CONFIG["model"]

    if not api_key:
        raise RuntimeError(
            "未找到 OPENAI_API_KEY。请设置环境变量 OPENAI_API_KEY，"
            "或在 config.json 中填写 api_key。"
        )

    return LLMConfig(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        model=model,
        temperature=float(merged.get("temperature", DEFAULT_CONFIG["temperature"])),
        max_tool_rounds=int(merged.get("max_tool_rounds", DEFAULT_CONFIG["max_tool_rounds"])),
        timeout_seconds=int(merged.get("timeout_seconds", DEFAULT_CONFIG["timeout_seconds"])),
    )

# -----------------------------
# Mock 工具实现
# -----------------------------

def query_city_weather(city: str, date: str = "今天", unit: str = "celsius") -> Dict[str, Any]:
    """Mock：查询指定城市的天气。"""
    normalized_city = city.strip()
    weather_db = {
        "北京": {"condition": "晴", "temperature_c": 28, "humidity": "36%", "wind": "东北风 2 级"},
        "上海": {"condition": "多云", "temperature_c": 27, "humidity": "68%", "wind": "东南风 3 级"},
        "杭州": {"condition": "小雨", "temperature_c": 24, "humidity": "82%", "wind": "西南风 2 级"},
        "广州": {"condition": "雷阵雨", "temperature_c": 30, "humidity": "78%", "wind": "南风 3 级"},
        "深圳": {"condition": "阵雨", "temperature_c": 29, "humidity": "75%", "wind": "东南风 3 级"},
        "成都": {"condition": "阴", "temperature_c": 23, "humidity": "70%", "wind": "微风"},
        "西安": {"condition": "晴", "temperature_c": 31, "humidity": "40%", "wind": "西北风 2 级"},
    }
    data = weather_db.get(
        normalized_city,
        {"condition": "多云", "temperature_c": 26, "humidity": "60%", "wind": "微风"},
    )

    temperature = data["temperature_c"]
    if unit == "fahrenheit":
        temperature = round(temperature * 9 / 5 + 32, 1)
        temperature_unit = "°F"
    else:
        temperature_unit = "°C"

    return {
        "city": normalized_city,
        "date": date,
        "condition": data["condition"],
        "temperature": temperature,
        "temperature_unit": temperature_unit,
        "humidity": data["humidity"],
        "wind": data["wind"],
        "source": "mock_weather_db",
    }

def query_city_attractions(city: str, preference: Optional[str] = None, top_k: int = 3) -> Dict[str, Any]:
    """Mock：查询指定城市的旅游景点。"""
    normalized_city = city.strip()
    attractions_db = {
        "北京": ["故宫博物院", "天坛公园", "颐和园", "八达岭长城", "什刹海"],
        "上海": ["外滩", "上海博物馆", "豫园", "东方明珠", "武康路"],
        "杭州": ["西湖", "灵隐寺", "西溪国家湿地公园", "良渚古城遗址公园", "河坊街"],
        "广州": ["陈家祠", "沙面岛", "广州塔", "越秀公园", "广东省博物馆"],
        "深圳": ["深圳湾公园", "莲花山公园", "大鹏所城", "华侨城创意文化园", "世界之窗"],
        "成都": ["武侯祠", "杜甫草堂", "宽窄巷子", "成都大熊猫繁育研究基地", "青城山"],
        "西安": ["秦始皇帝陵博物院", "西安城墙", "大雁塔", "陕西历史博物馆", "华清宫"],
    }
    attractions = attractions_db.get(normalized_city, ["城市博物馆", "老街区", "中心公园", "当地美食街", "历史文化街区"])
    top_k = max(1, min(int(top_k), len(attractions)))

    return {
        "city": normalized_city,
        "preference": preference or "未指定",
        "attractions": attractions[:top_k],
        "tips": "以上为 mock 数据，适合用于验证模型是否会主动调用工具。",
        "source": "mock_attractions_db",
    }

TOOL_FUNCTIONS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "query_city_weather": query_city_weather,
    "query_city_attractions": query_city_attractions,
}

# -----------------------------
# OpenAI tools / functions schema
# -----------------------------

FUNCTION_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "query_city_weather",
        "description": "查询某个城市的 mock 天气信息，包括天气状况、温度、湿度和风力。",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称，例如：北京、上海、杭州。"},
                "date": {"type": "string", "description": "查询日期，例如：今天、明天、周末。默认今天。"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "温度单位，默认 celsius。",
                },
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
    {
        "name": "query_city_attractions",
        "description": "查询某个城市的 mock 旅游景点列表，可按偏好返回若干推荐。",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称，例如：北京、上海、杭州。"},
                "preference": {
                    "type": "string",
                    "description": "旅行偏好，例如：历史文化、自然风光、亲子、美食、citywalk。",
                },
                "top_k": {"type": "integer", "description": "返回景点数量，默认 3。", "minimum": 1, "maximum": 5},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
]

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {"type": "function", "function": function_definition}
    for function_definition in FUNCTION_DEFINITIONS
]

class OpenAIProtocolAgent:
    """一个基于 OpenAI Chat Completions 协议的轻量 Agent。"""

    def __init__(self, config: LLMConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.endpoint = f"{self.config.base_url}/chat/completions"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            self.endpoint,
            headers=self._headers(),
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"LLM 请求失败：HTTP {response.status_code}\n{response.text}")

        data = response.json()
        if "choices" not in data:
            raise RuntimeError(
                "LLM 响应中没有 choices 字段。通常原因是模型名不可用、服务端不兼容该协议字段，"
                f"或返回了错误对象。原始响应：\n{json.dumps(data, ensure_ascii=False, indent=2)}"
            )
        return data

    @staticmethod
    def _safe_json_loads(raw: Optional[str]) -> Dict[str, Any]:
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # 某些模型可能返回非严格 JSON；这里直接抛错，便于测试时暴露问题。
            raise ValueError(f"工具参数不是合法 JSON：{raw}")

    @staticmethod
    def _coerce_parameter_value(value: str) -> Any:
        """把类 XML 工具调用中的字符串参数尽量转换为更合适的 Python 类型。"""
        stripped = value.strip()
        if stripped.lower() in {"true", "false"}:
            return stripped.lower() == "true"
        if re.fullmatch(r"[-+]?\d+", stripped):
            return int(stripped)
        if re.fullmatch(r"[-+]?(\d+\.\d*|\d*\.\d+)", stripped):
            return float(stripped)
        return stripped

    @classmethod
    def _parse_xml_like_tool_calls(cls, text: Optional[str]) -> List[Dict[str, Any]]:
        """解析部分模型在 reasoning_content/content 中输出的非标准类 XML 工具调用。

        例如：
            <tool_call>
            <function=query_city_weather>
            <parameter=city>杭州</parameter>
            </function>
            </tool_call>

        这不是 OpenAI 标准 tool_calls 结构，但一些 OpenAI 兼容服务会这样返回。
        为了让 Agent 能继续测试工具调用能力，这里把它转换成标准 tool_calls 形态。
        """
        if not text:
            return []

        tool_calls: List[Dict[str, Any]] = []
        tool_blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL)
        for block in tool_blocks:
            function_match = re.search(r"<function=([^>]+)>\s*(.*?)\s*</function>", block, flags=re.DOTALL)
            if not function_match:
                continue

            function_name = function_match.group(1).strip()
            function_body = function_match.group(2)
            arguments: Dict[str, Any] = {}

            parameter_matches = re.findall(
                r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
                function_body,
                flags=re.DOTALL,
            )
            for parameter_name, parameter_value in parameter_matches:
                arguments[parameter_name.strip()] = cls._coerce_parameter_value(parameter_value)

            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                    "_compat_source": "reasoning_content_xml",
                }
            )

        return tool_calls

    def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name not in TOOL_FUNCTIONS:
            return {"error": f"未知工具：{name}"}
        try:
            result = TOOL_FUNCTIONS[name](**arguments)
            return {"ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": repr(exc)}

    def run_with_tools(self, question: str) -> str:
        """使用现代 OpenAI tools / tool_calls 协议运行。"""
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "你是一个旅行助手。只要用户询问天气或旅游景点，就必须优先调用可用工具获取信息，"
                    "然后基于工具结果用中文给出简洁、准确的回答。"
                ),
            },
            {"role": "user", "content": question},
        ]

        for round_index in range(self.config.max_tool_rounds):
            payload = {
                "model": self.config.model,
                "messages": messages,
                "tools": TOOL_DEFINITIONS,
                "tool_choice": "auto",
                "temperature": self.config.temperature,
            }
            data = self._chat(payload)
            print(data)
            message = data["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []

            # 兼容非标准返回：部分模型不会填充 OpenAI 标准 tool_calls 字段，
            # 而是把 <tool_call>...</tool_call> 放在 reasoning_content 或 content 中。
            compat_tool_calls = False
            if not tool_calls:
                tool_calls = self._parse_xml_like_tool_calls(message.get("reasoning_content"))
                if not tool_calls:
                    tool_calls = self._parse_xml_like_tool_calls(message.get("content"))
                compat_tool_calls = bool(tool_calls)

            if not tool_calls:
                if self.verbose:
                    print(f"[tools] 第 {round_index + 1} 轮：模型未继续调用工具，输出最终答案。", file=sys.stderr)
                return message.get("content") or message.get("reasoning_content") or ""

            if compat_tool_calls:
                # 给后续请求补一个标准 assistant tool_calls 消息，便于 OpenAI 兼容服务接收 role=tool 结果。
                message = {
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": [
                        {key: value for key, value in tool_call.items() if not key.startswith("_")}
                        for tool_call in tool_calls
                    ],
                }

            messages.append(message)
            if self.verbose:
                source = "非标准 reasoning_content/content" if compat_tool_calls else "标准 tool_calls"
                print(
                    f"[tools] 第 {round_index + 1} 轮：模型通过{source}请求调用 {len(tool_calls)} 个工具。",
                    file=sys.stderr,
                )

            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                name = function.get("name", "")
                arguments = self._safe_json_loads(function.get("arguments"))
                result = self._execute_tool(name, arguments)

                if self.verbose:
                    print(f"[tools] 调用工具：{name}({json.dumps(arguments, ensure_ascii=False)})", file=sys.stderr)
                    print(f"[tools] 工具结果：{json.dumps(result, ensure_ascii=False)}", file=sys.stderr)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        raise RuntimeError("达到最大工具调用轮次，模型仍未给出最终答案。")

    def run_with_functions(self, question: str) -> str:
        """使用旧版 OpenAI functions / function_call 协议运行。"""
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "你是一个旅行助手。只要用户询问天气或旅游景点，就必须优先调用函数获取信息，"
                    "然后基于函数结果用中文给出简洁、准确的回答。"
                ),
            },
            {"role": "user", "content": question},
        ]

        for round_index in range(self.config.max_tool_rounds):
            payload = {
                "model": self.config.model,
                "messages": messages,
                "functions": FUNCTION_DEFINITIONS,
                "function_call": "auto",
                "temperature": self.config.temperature,
            }
            data = self._chat(payload)
            print(data)
            message = data["choices"][0]["message"]
            function_call = message.get("function_call")

            if not function_call:
                if self.verbose:
                    print(f"[functions] 第 {round_index + 1} 轮：模型未继续调用函数，输出最终答案。", file=sys.stderr)
                return message.get("content") or ""

            name = function_call.get("name", "")
            arguments = self._safe_json_loads(function_call.get("arguments"))
            result = self._execute_tool(name, arguments)

            if self.verbose:
                print(f"[functions] 第 {round_index + 1} 轮：模型请求调用函数。", file=sys.stderr)
                print(f"[functions] 调用函数：{name}({json.dumps(arguments, ensure_ascii=False)})", file=sys.stderr)
                print(f"[functions] 函数结果：{json.dumps(result, ensure_ascii=False)}", file=sys.stderr)

            messages.append(message)
            messages.append(
                {
                    "role": "function",
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

        raise RuntimeError("达到最大函数调用轮次，模型仍未给出最终答案。")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试 OpenAI 协议模型的 tool/function calling 能力。")
    parser.add_argument(
        "--question",
        default="我明天去杭州，天气怎么样？请再推荐 3 个适合初次游玩的景点。",
        help="用户问题。",
    )
    parser.add_argument(
        "--mode",
        choices=["tools", "functions"],
        default="tools",
        help="调用协议模式：tools 为现代 tool calling；functions 为旧版 function calling。",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="配置文件路径。",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="不打印工具调用过程，只输出最终答案。",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    agent = OpenAIProtocolAgent(config=config, verbose=not args.quiet)

    if args.mode == "tools":
        answer = agent.run_with_tools(args.question)
    else:
        answer = agent.run_with_functions(args.question)

    print("\n===== FINAL ANSWER =====")
    print(answer)

if __name__ == "__main__":
    main()
