import json
from typing import Any, Optional

from datasets import DatasetDict, concatenate_datasets, load_dataset

from preprocessing.mappers.utils import row_mapper_to_batched

DATA_SOURCE = "nvidia/Nemotron-Competitive-Programming-v1"

_INFINIBYTE_PARTS = [
    ("infinibyte_part_00", "hf://datasets/nvidia/Nemotron-Competitive-Programming-v1/data/infinibyte.part_00.jsonl"),
    ("infinibyte_part_01", "hf://datasets/nvidia/Nemotron-Competitive-Programming-v1/data/infinibyte.part_01.jsonl"),
]
_SAMPLE_SIZE = 50_000
_SEED = 42


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


def load_nemotron_cp() -> DatasetDict:
    """Load infinibyte parts, filter used_in==['nano_v3'], sample 50k, track origin."""
    part_datasets = []
    part_boundaries = [0]
    for _, data_file in _INFINIBYTE_PARTS:
        part = load_dataset("json", data_files={"train": data_file}, split="train")
        part_datasets.append(part)
        part_boundaries.append(part_boundaries[-1] + len(part))

    ds = concatenate_datasets(part_datasets)
    ds = ds.add_column("_concat_idx", list(range(len(ds))))
    ds = ds.filter(lambda x: x["used_in"] == ["nano_v3"])
    ds = ds.shuffle(seed=_SEED).select(range(min(_SAMPLE_SIZE, len(ds))))

    original_splits: list[str] = []
    original_indices: list[int] = []
    for concat_idx in ds["_concat_idx"]:
        found = False
        for i in range(len(_INFINIBYTE_PARTS) - 1, -1, -1):
            if concat_idx >= part_boundaries[i]:
                original_splits.append(_INFINIBYTE_PARTS[i][0])
                original_indices.append(concat_idx - part_boundaries[i])
                found = True
                break
        if not found:
            original_splits.append(_INFINIBYTE_PARTS[0][0])
            original_indices.append(concat_idx)

    ds = ds.add_column("_original_split", original_splits)
    ds = ds.add_column("_original_index", original_indices)
    return DatasetDict({"train": ds})


def _map_nemotron_cp_row(example: dict[str, Any], idx: int) -> Optional[dict[str, Any]]:
    problem = _extract_first_message(example["messages"], "user")
    if problem is None:
        return None

    answer = _extract_first_message(example["messages"], "assistant")
    reference: dict[str, Any] = {}
    if answer is not None:
        reference["expert_solution"] = answer

    return {
        "prompt": [{"role": "user", "content": problem}],
        "reference": json.dumps(reference, ensure_ascii=True),
        "data_source": DATA_SOURCE,
        "meta_information": json.dumps(
            {
                "original_split": example.get("_original_split"),
                "original_index": example.get("_original_index"),
            },
            ensure_ascii=True,
        ),
        "data_source_id": example.get("uuid", str(idx)),
        "turn": 0,
    }


map_nemotron_cp = row_mapper_to_batched(_map_nemotron_cp_row)
