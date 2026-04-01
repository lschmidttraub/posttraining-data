import json
from typing import Any, Optional

from datasets import DatasetDict, load_dataset

from preprocessing.mappers.utils import row_mapper_to_batched

DATA_SOURCE = "open-r1/DAPO-Math-17k-Processed"

MATH_INSTRUCTION_PREFIX = (
    "{question}\n\n"
    "Please answer step by step, and put your final answer within \\boxed{{}}.\n"
)


def load_dapo_math() -> DatasetDict:
    ds = load_dataset(DATA_SOURCE, "en", split="train")
    return DatasetDict({"train": ds})


def _map_dapo_math_row(example: dict[str, Any], idx: int) -> Optional[dict[str, Any]]:
    return {
        "prompt": [{"role": "user", "content": MATH_INSTRUCTION_PREFIX.format(question=example["prompt"])}],
        "reference": json.dumps({"expected_answer": example["solution"]}, ensure_ascii=True),
        "data_source": DATA_SOURCE,
        "meta_information": json.dumps({"split": "en"}, ensure_ascii=True),
        "data_source_id": str(idx),
        "turn": 0,
    }


map_dapo_math = row_mapper_to_batched(_map_dapo_math_row)
