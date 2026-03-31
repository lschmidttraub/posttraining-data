import json
from typing import Any, Optional

from preprocessing.mappers.utils import row_mapper_to_batched

DATA_SOURCE = "zwhe99/DeepMath-103K"

MATH_INSTRUCTION_PREFIX = (
    "Solve the following math problem. Make sure to put the answer "
    "(and only the answer) inside \\boxed{}.\n\n"
)


def _map_deepmath_row(example: dict[str, Any], idx: int) -> Optional[dict[str, Any]]:
    reference: dict[str, Any] = {"expected_answer": example["final_answer"]}
    if example.get("r1_solution_1"):
        reference["expert_solution"] = example["r1_solution_1"]

    return {
        "prompt": [{"role": "user", "content": MATH_INSTRUCTION_PREFIX + example["question"]}],
        "reference": json.dumps(reference, ensure_ascii=True),
        "data_source": DATA_SOURCE,
        "meta_information": json.dumps(
            {
                "difficulty": example.get("difficulty"),
                "topic": example.get("topic"),
            },
            ensure_ascii=True,
        ),
        "data_source_id": str(idx),
        "turn": 0,
    }


map_deepmath = row_mapper_to_batched(_map_deepmath_row)
