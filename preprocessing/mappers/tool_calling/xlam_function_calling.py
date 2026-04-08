import json
from typing import Any

from preprocessing.mappers.utils import normalize_argument_value, safe_json_loads


DATA_SOURCE = "Salesforce/xlam-function-calling-60k"

XLAM_FUNCTION_CALLING_PROMPT_TEMPLATE = '''You are a helpful assistant with access to tools.

When the user's request requires tool calls, work through these steps:

1. **Analyze the request**: Identify what the user needs and which parts require tool calls.
2. **Select tools**: Determine which available tool(s) match the task. Explain why each tool is appropriate.
3. **Construct arguments**: Map the user's requirements to the tool's parameters, noting any defaults or transformations needed.
4. **Make the calls**: Respond with a JSON array of tool calls. Each tool call must be an object with "name" and "arguments" keys.

If the request can be answered without tools, respond with a normal assistant message.
Do not include code fences around the JSON — only the raw JSON array.

### Example

User: "What's the weather like in Paris and Tokyo?"

Step 1: The user wants current weather for two cities.
Step 2: The `get_weather` tool retrieves weather by location — I need to call it once per city.
Step 3: The required parameter is `location` (string). "Paris" and "Tokyo" map directly.

[{{"name": "get_weather", "arguments": {{"location": "Paris"}}}}, {{"name": "get_weather", "arguments": {{"location": "Tokyo"}}}}]

Available tools:
{tools}
'''


def _format_tool_prompt(tools: list[dict[str, Any]], query: str) -> list[dict[str, str]]:
    system_content = XLAM_FUNCTION_CALLING_PROMPT_TEMPLATE.format(tools=json.dumps(tools, ensure_ascii=True, indent=2))
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
    ]


def _normalize_answers(answers: Any) -> list[dict[str, Any]]:
    normalized = []
    if isinstance(answers, list):
        for answer in answers:
            if not isinstance(answer, dict):
                continue
            normalized.append(
                {
                    "name": answer.get("name", ""),
                    "arguments": {
                        key: normalize_argument_value(value)
                        for key, value in (answer.get("arguments", {}) or {}).items()
                    },
                }
            )
    return normalized


def map_xlam_function_calling(
    batch: dict[str, list[Any]], indices: list[int]
) -> dict[str, list[Any]]:
    prompts = []
    references = []
    data_sources = []
    data_source_ids = []
    meta_information = []
    turns = []

    ids = batch.get("id", [])
    queries = batch.get("query", [])
    tools_batch = batch.get("tools", [])
    answers_batch = batch.get("answers", [])

    for offset, row_idx in enumerate(indices):
        query = queries[offset] if offset < len(queries) and queries[offset] is not None else ""
        tools = safe_json_loads(tools_batch[offset] if offset < len(tools_batch) else "[]", [])
        answers = safe_json_loads(answers_batch[offset] if offset < len(answers_batch) else "[]", [])

        normalized_answers = _normalize_answers(answers)

        prompts.append(_format_tool_prompt(tools if isinstance(tools, list) else [], query))
        references.append(json.dumps(normalized_answers, ensure_ascii=True))
        data_sources.append(DATA_SOURCE)
        data_source_ids.append(str(ids[offset] if offset < len(ids) and ids[offset] is not None else row_idx))
        meta_information.append(
            json.dumps(
                {
                    "query": query,
                    "tools": tools if isinstance(tools, list) else [],
                    "answer_format": "json_function_calls",
                    "num_tools": len(tools) if isinstance(tools, list) else 0,
                    "num_reference_calls": len(normalized_answers),
                },
                ensure_ascii=True,
            )
        )
        turns.append(0)

    return {
        "prompt": prompts,
        "reference": references,
        "data_source": data_sources,
        "data_source_id": data_source_ids,
        "meta_information": meta_information,
        "turn": turns,
    }
