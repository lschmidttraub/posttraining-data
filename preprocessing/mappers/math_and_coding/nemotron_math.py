import json
from typing import Any, Optional

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from preprocessing.mappers.utils import row_mapper_to_batched

DATA_SOURCE = "nvidia/Nemotron-Math-v2"

MATH_INSTRUCTION_PREFIX = (
    "{question}\n\n"
    "Please answer step by step, and put your final answer within \\boxed{{}}.\n"
)

SPLIT_PRIORITY = {
    "high_part00": 0,
    "high_part01": 0,
    "high_part02": 0,
    "medium": 1,
    "low": 2,
}


def _dedupe_by_problem(ds_dict: DatasetDict) -> Dataset:
    """Pick one row per unique problem with priority: high > medium > low."""
    seen: dict[str, str] = {}
    indices_per_split: dict[str, list[int]] = {s: [] for s in ds_dict}

    for split in sorted(ds_dict.keys(), key=lambda s: SPLIT_PRIORITY.get(s, 99)):
        problems = ds_dict[split]["problem"]
        for i, p in enumerate(problems):
            if p not in seen:
                seen[p] = split
                indices_per_split[split].append(i)

    subsets = []
    for split, idxs in indices_per_split.items():
        if idxs:
            subset = ds_dict[split].select(idxs)
            subset = subset.add_column("_source_split", [split] * len(subset))
            subset = subset.add_column("_original_idx", idxs)
            subsets.append(subset)

    return concatenate_datasets(subsets)


def load_nemotron_math() -> DatasetDict:
    """Load all splits, deduplicate by problem with priority, return flat."""
    ds_dict = load_dataset(DATA_SOURCE)
    deduped = _dedupe_by_problem(ds_dict)
    return DatasetDict({"train": deduped})


def _map_nemotron_math_row(example: dict[str, Any], idx: int) -> Optional[dict[str, Any]]:
    meta_raw = example.get("metadata")
    if isinstance(meta_raw, str):
        meta_raw = json.loads(meta_raw)

    source_split = example.get("_source_split", "")
    original_idx = example.get("_original_idx", idx)

    messages = example.get("messages", [])
    expert_solution = None
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                expert_solution = msg["content"]
                break

    reference: dict[str, Any] = {"expected_answer": example["expected_answer"]}
    if expert_solution is not None:
        reference["expert_solution"] = expert_solution

    return {
        "prompt": [{"role": "user", "content": MATH_INSTRUCTION_PREFIX.format(question=example["problem"])}],
        "reference": json.dumps(reference, ensure_ascii=True),
        "data_source": DATA_SOURCE,
        "meta_information": json.dumps(
            {
                "changed_answer_to_majority": example.get("changed_answer_to_majority"),
                "original_data_source": example.get("data_source"),
                "source_split": source_split,
                "difficulty": meta_raw,
                "tool": example.get("tool"),
                "url": example.get("url"),
            },
            ensure_ascii=True,
        ),
        "data_source_id": f"{source_split}/{original_idx}",
        "turn": 0,
    }


map_nemotron_math = row_mapper_to_batched(_map_nemotron_math_row)
