import json
from typing import Any

from preprocessing.mappers.tool_calling.common import (
    extract_text_content,
    format_tool_prompt,
    make_tool_output_message,
    serialize_tool_calls,
    stringify_content,
)
from preprocessing.mappers.utils import safe_json_loads


DATA_SOURCE = "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1"


def _normalize_input_item(item: Any) -> dict[str, str] | None:
    if not isinstance(item, dict):
        return None

    item_type = stringify_content(item.get("type", "")).strip().lower()
    role = stringify_content(item.get("role", "")).strip().lower()

    if item_type == "reasoning":
        return None
    if item_type == "function_call":
        return {
            "role": "assistant",
            "content": serialize_tool_calls([item]),
        }
    if item_type == "function_call_output":
        return make_tool_output_message(item.get("output", ""))

    if role in {"system", "user", "assistant"}:
        content = extract_text_content(item.get("content", ""))
        return {"role": role, "content": content}

    return None


def _normalize_expected_action(expected_action: Any) -> tuple[str, str]:
    expected_action = safe_json_loads(expected_action, expected_action)

    if isinstance(expected_action, dict):
        action_type = stringify_content(expected_action.get("type", "")).strip().lower()
        role = stringify_content(expected_action.get("role", "")).strip().lower()

        if action_type == "function_call":
            return serialize_tool_calls([expected_action]), "tool_call"
        if action_type == "message" or role == "assistant":
            return extract_text_content(expected_action.get("content", "")), "message"

    if isinstance(expected_action, str):
        return expected_action, "text"

    return stringify_content(expected_action), "unknown"


def map_nemotron_rl_agentic_conversational_tool_use_pivot(
    batch: dict[str, list[Any]], indices: list[int]
) -> dict[str, list[Any]]:
    prompts: list[list[dict[str, str]]] = []
    references: list[str] = []
    data_sources: list[str] = []
    data_source_ids: list[str] = []
    meta_information: list[str] = []
    turns: list[int] = []

    trajectory_ids = batch.get("trajectory_id", [])
    params_batch = batch.get("responses_create_params", [])
    expected_actions = batch.get("expected_action", [])
    scenarios = batch.get("scenario", [])
    num_unique_actions = batch.get("num_unique_actions", [])
    meta_infos = batch.get("meta_info", [])
    pass_rates = batch.get("pass_rate", [])
    pass_rate_totals = batch.get("pass_rate_total", [])
    pass_rate_passed = batch.get("pass_rate_passed", [])

    for offset, row_idx in enumerate(indices):
        params = safe_json_loads(params_batch[offset] if offset < len(params_batch) else {}, {})
        if not isinstance(params, dict):
            continue

        input_items = params.get("input", [])
        tools = params.get("tools", [])
        if not isinstance(input_items, list) or not isinstance(tools, list):
            continue

        prompt_messages = [message for item in input_items if (message := _normalize_input_item(item)) is not None]
        reference, reference_type = _normalize_expected_action(
            expected_actions[offset] if offset < len(expected_actions) else ""
        )
        meta_info = safe_json_loads(meta_infos[offset] if offset < len(meta_infos) else {}, {})

        turn_index = meta_info.get("turn", 0) if isinstance(meta_info, dict) else 0
        step_index = meta_info.get("step") if isinstance(meta_info, dict) else None
        trajectory_id = trajectory_ids[offset] if offset < len(trajectory_ids) else row_idx
        data_source_id = str(trajectory_id)
        if step_index is not None:
            data_source_id = f"{data_source_id}:step{step_index}"

        prompts.append(format_tool_prompt(tools, prompt_messages))
        references.append(reference)
        data_sources.append(DATA_SOURCE)
        data_source_ids.append(data_source_id)
        meta_information.append(
            json.dumps(
                {
                    "scenario": scenarios[offset] if offset < len(scenarios) else None,
                    "num_unique_actions": num_unique_actions[offset] if offset < len(num_unique_actions) else None,
                    "meta_info": meta_info if isinstance(meta_info, dict) else meta_infos[offset] if offset < len(meta_infos) else None,
                    "reference_type": reference_type,
                    "num_tools": len(tools),
                    "num_prompt_messages": len(prompt_messages),
                    "pass_rate": pass_rates[offset] if offset < len(pass_rates) else None,
                    "pass_rate_total": pass_rate_totals[offset] if offset < len(pass_rate_totals) else None,
                    "pass_rate_passed": pass_rate_passed[offset] if offset < len(pass_rate_passed) else None,
                },
                ensure_ascii=True,
            )
        )
        turns.append(turn_index if isinstance(turn_index, int) else 0)

    return {
        "prompt": prompts,
        "reference": references,
        "data_source": data_sources,
        "data_source_id": data_source_ids,
        "meta_information": meta_information,
        "turn": turns,
    }
