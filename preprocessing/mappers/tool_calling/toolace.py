import json
from typing import Any

from preprocessing.mappers.tool_calling.common import TOOL_CALLING_STEP_BY_STEP_PREAMBLE, stringify_content
from preprocessing.mappers.utils import safe_json_loads


DATA_SOURCE = "Team-ACE/ToolACE"


def _normalize_toolace_message(message: Any) -> dict[str, str] | None:
    if not isinstance(message, dict):
        return None

    raw_role = stringify_content(message.get("from", message.get("role", ""))).strip().lower()
    content = stringify_content(message.get("value", message.get("content", "")))
    if not content:
        return None

    if raw_role in {"assistant", "gpt"}:
        return {"role": "assistant", "content": content}
    if raw_role in {"tool", "function", "observation"}:
        return {"role": "user", "content": f"Tool output:\n{content}"}
    return {"role": "user", "content": content}


def map_toolace(batch: dict[str, list[Any]], indices: list[int]) -> dict[str, list[Any]]:
    prompts: list[list[dict[str, str]]] = []
    references: list[str] = []
    data_sources: list[str] = []
    data_source_ids: list[str] = []
    meta_information: list[str] = []
    turns: list[int] = []

    ids = batch.get("id", [])
    system_prompts = batch.get("system", [])
    conversations_batch = batch.get("conversations", [])

    for offset, row_idx in enumerate(indices):
        base_id = str(ids[offset] if offset < len(ids) and ids[offset] is not None else row_idx)
        system_prompt = stringify_content(system_prompts[offset] if offset < len(system_prompts) else "")
        conversations = safe_json_loads(conversations_batch[offset] if offset < len(conversations_batch) else [], [])
        if not isinstance(conversations, list):
            continue

        history: list[dict[str, str]] = []
        assistant_turn = 0
        for message_idx, raw_message in enumerate(conversations):
            normalized_message = _normalize_toolace_message(raw_message)
            if normalized_message is None:
                continue

            if normalized_message["role"] == "assistant":
                prompt_messages = list(history)
                if system_prompt:
                    augmented_system = system_prompt.rstrip() + "\n" + TOOL_CALLING_STEP_BY_STEP_PREAMBLE
                    prompt_messages = [{"role": "system", "content": augmented_system}, *prompt_messages]

                prompts.append(prompt_messages)
                references.append(normalized_message["content"])
                data_sources.append(DATA_SOURCE)
                data_source_ids.append(f"{base_id}:turn{assistant_turn}")
                meta_information.append(
                    json.dumps(
                        {
                            "conversation_length": len(conversations),
                            "assistant_turn_index": assistant_turn,
                            "source_message_index": message_idx,
                            "has_system_prompt": bool(system_prompt),
                        },
                        ensure_ascii=True,
                    )
                )
                turns.append(assistant_turn)
                assistant_turn += 1

            history.append(normalized_message)

    return {
        "prompt": prompts,
        "reference": references,
        "data_source": data_sources,
        "data_source_id": data_source_ids,
        "meta_information": meta_information,
        "turn": turns,
    }
