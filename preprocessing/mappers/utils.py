import json
from typing import Any


def safe_json_loads(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value if value is not None else default


def normalize_argument_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, dict)):
        return value
    return str(value)


def format_tool_prompt(tools: list[dict[str, Any]], query: str) -> list[dict[str, str]]:
    system_content = (
        "You are a helpful assistant with access to tools. "
        "Return the correct tool call or tool calls for the user's request.\n\n"
        "Available tools:\n"
        f"{json.dumps(tools, ensure_ascii=True, indent=2)}"
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
    ]
