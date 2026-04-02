import json
from typing import Any

from preprocessing.mappers.utils import inject_system_prompt
from preprocessing.system_prompts import SYSTEM_PROMPT_SCIENCE


DATA_SOURCE = "nvidia/Nemotron-Science-v1"


def _normalize_message(message: dict[str, Any]) -> dict[str, str]:
    return {
        "role": str(message["role"]),
        "content": str(message["content"]),
    }


def map_nemotron_science(
    batch: dict[str, list[Any]], indices: list[int]
) -> dict[str, list[Any]]:
    """Maps both MCQ and RQA subsets of Nemotron-Science-v1.

    Each row has a `messages` field with exactly two entries: [user, assistant].
    The subset name is not available from the raw columns, so it is omitted;
    register this mapper separately per subset if subset tracking is needed.
    """
    prompts = []
    references = []
    data_sources = []
    data_source_ids = []
    meta_information = []
    turns = []

    messages_batch = batch.get("messages", [])
    uuids = batch.get("uuid", [])
    licenses = batch.get("license", [])
    used_in_batch = batch.get("used_in", [])
    tools_batch = batch.get("tools", [])

    for offset, row_idx in enumerate(indices):
        messages = messages_batch[offset] if offset < len(messages_batch) else []
        uuid = uuids[offset] if offset < len(uuids) else str(row_idx)
        license_val = licenses[offset] if offset < len(licenses) else None
        used_in = used_in_batch[offset] if offset < len(used_in_batch) else []
        tools = tools_batch[offset] if offset < len(tools_batch) else []

        if len(messages) != 2:
            continue

        user_message, assistant_message = messages
        if user_message.get("role") != "user" or assistant_message.get("role") != "assistant":
            continue

        prompts.append(
            inject_system_prompt(
                [_normalize_message(user_message)],
                SYSTEM_PROMPT_SCIENCE,
            )
        )
        references.append(json.dumps({}, ensure_ascii=True))
        data_sources.append(DATA_SOURCE)
        data_source_ids.append(str(uuid))
        meta_information.append(
            json.dumps(
                {
                    "responses": [_normalize_message(assistant_message)],
                    "license": license_val,
                    "used_in": used_in if used_in else [],
                    "tools": tools if tools else [],
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
