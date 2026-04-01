import json
import re
from typing import Any

from preprocessing.mappers.tool_calling.common import format_tool_prompt, stringify_content
from preprocessing.mappers.utils import safe_json_loads


DATA_SOURCE = "nvidia/When2Call"
DATASET_CONFIG = "train_sft"

TOOLCALL_PATTERN = re.compile(r"^\s*<TOOLCALL>(.*)</TOOLCALL>\s*$", re.DOTALL)


def _normalize_response_text(content: Any) -> str:
    text = stringify_content(content)
    match = TOOLCALL_PATTERN.match(text)
    if match:
        return match.group(1).strip()
    return text


def _normalize_message(message: Any) -> dict[str, str] | None:
    if not isinstance(message, dict):
        return None

    role = stringify_content(message.get("role", "user")).strip().lower() or "user"
    if role not in {"system", "user", "assistant"}:
        role = "user"

    return {
        "role": role,
        "content": _normalize_response_text(message.get("content", "")),
    }


def map_when2call_train_sft(
    batch: dict[str, list[Any]], indices: list[int]
) -> dict[str, list[Any]]:
    prompts: list[list[dict[str, str]]] = []
    references: list[str] = []
    data_sources: list[str] = []
    data_source_ids: list[str] = []
    meta_information: list[str] = []
    turns: list[int] = []

    uuids = batch.get("uuid", [])
    tools_batch = batch.get("tools", [])
    messages_batch = batch.get("messages", [])

    for offset, row_idx in enumerate(indices):
        tools = safe_json_loads(tools_batch[offset] if offset < len(tools_batch) else [], [])
        messages = safe_json_loads(messages_batch[offset] if offset < len(messages_batch) else [], [])
        if not isinstance(tools, list) or not isinstance(messages, list) or not messages:
            continue

        normalized_messages = [message for item in messages if (message := _normalize_message(item)) is not None]
        if not normalized_messages:
            continue

        reference_message = normalized_messages[-1]
        history_messages = normalized_messages[:-1]

        prompts.append(format_tool_prompt(tools, history_messages))
        references.append(reference_message["content"])
        data_sources.append(DATA_SOURCE)
        data_source_ids.append(str(uuids[offset] if offset < len(uuids) and uuids[offset] is not None else row_idx))
        meta_information.append(
            json.dumps(
                {
                    "config": DATASET_CONFIG,
                    "tools": tools,
                    "reference_role": reference_message["role"],
                    "reference_is_tool_call": bool(TOOLCALL_PATTERN.match(stringify_content(messages[-1].get("content", "")) if isinstance(messages[-1], dict) else "")),
                    "num_tools": len(tools),
                    "num_messages": len(normalized_messages),
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
