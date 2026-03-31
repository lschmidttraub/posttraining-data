import json
from typing import Any, List, Optional

from datasets import DatasetDict, load_dataset

from preprocessing.mappers.utils import row_mapper_to_batched

DATA_SOURCE = "nvidia/Nemotron-Cascade-SFT-Stage-1"


def _extract_first_message(messages: Any, role: str) -> Optional[str]:
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            return None
    if not isinstance(messages, list):
        return None
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == role:
            return msg.get("content")
    return None


def load_nemotron_cascade_code() -> DatasetDict:
    """Load code config, deduplicate by user prompt, return as DatasetDict."""
    ds = load_dataset(DATA_SOURCE, "code", split="train")

    all_messages = ds["messages"]
    seen: set[str] = set()
    unique_indices: List[int] = []
    for idx, messages in enumerate(all_messages):
        problem = _extract_first_message(messages, "user")
        if problem is not None and problem not in seen:
            seen.add(problem)
            unique_indices.append(idx)

    deduped = ds.select(unique_indices)
    deduped = deduped.add_column("_original_idx", unique_indices)
    return DatasetDict({"train": deduped})


def _map_nemotron_cascade_code_row(
    example: dict[str, Any], idx: int
) -> Optional[dict[str, Any]]:
    problem = _extract_first_message(example["messages"], "user")
    if problem is None:
        return None

    answer = _extract_first_message(example["messages"], "assistant")
    reference: dict[str, Any] = {}
    if answer is not None:
        reference["expert_solution"] = answer

    original_idx = example.get("_original_idx", idx)

    return {
        "prompt": [{"role": "user", "content": problem}],
        "reference": json.dumps(reference, ensure_ascii=True),
        "data_source": DATA_SOURCE,
        "meta_information": json.dumps(
            {
                "original_data_source": example.get("source"),
                "expert_generator_model": example.get("generator"),
            },
            ensure_ascii=True,
        ),
        "data_source_id": str(original_idx),
        "turn": 0,
    }


map_nemotron_cascade_code = row_mapper_to_batched(_map_nemotron_cascade_code_row)
