import json
from typing import Any

from preprocessing.mappers.utils import normalize_argument_value, safe_json_loads


TOOL_CALLING_CHAT_PROMPT_TEMPLATE = """You are a helpful assistant with access to tools.
When the user's request requires tool calls, respond with a JSON array of tool calls.
Each tool call must be an object with "name" and "arguments" keys.
If the request can be answered without tools, respond with a normal assistant message.
Do not include code fences.

Available tools:
{tools}
"""


def stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True)
    except TypeError:
        return str(value)


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if not isinstance(item, dict):
                continue
            if "text" in item and item["text"] is not None:
                chunks.append(stringify_content(item["text"]))
        return "".join(chunks)
    if isinstance(content, dict):
        if "text" in content:
            return stringify_content(content.get("text"))
        return stringify_content(content)
    return stringify_content(content)


def normalize_tool_schema(tool: Any) -> dict[str, Any]:
    tool = safe_json_loads(tool, {})
    if not isinstance(tool, dict):
        return {}

    if tool.get("type") == "function":
        return {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {}) or {},
        }

    return {
        "name": tool.get("name", ""),
        "description": tool.get("description", ""),
        "parameters": tool.get("parameters", {}) or {},
    }


def format_tool_prompt(
    tools: list[dict[str, Any]],
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    system_content = TOOL_CALLING_CHAT_PROMPT_TEMPLATE.format(
        tools=json.dumps([normalize_tool_schema(tool) for tool in tools], ensure_ascii=True, indent=2)
    )
    return [{"role": "system", "content": system_content}, *messages]


def serialize_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    normalized_calls: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        raw_arguments = safe_json_loads(tool_call.get("arguments", {}), {})
        arguments = raw_arguments if isinstance(raw_arguments, dict) else {}
        normalized_calls.append(
            {
                "name": stringify_content(tool_call.get("name", "")).strip(),
                "arguments": {
                    key: normalize_argument_value(value)
                    for key, value in arguments.items()
                },
            }
        )
    return json.dumps(normalized_calls, ensure_ascii=True)


def make_tool_output_message(output: Any) -> dict[str, str]:
    return {
        "role": "user",
        "content": f"Tool output:\n{extract_text_content(output)}",
    }
