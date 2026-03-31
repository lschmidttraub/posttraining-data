import json
from typing import Any, Optional

from preprocessing.mappers.utils import row_mapper_to_batched

DATA_SOURCE = "nlile/NuminaMath-1.5-RL-Verifiable"

MATH_INSTRUCTION_PREFIX = (
    "Solve the following math problem. Make sure to put the answer "
    "(and only the answer) inside \\boxed{}.\n\n"
)


def _map_numina_math_row(example: dict[str, Any], idx: int) -> Optional[dict[str, Any]]:
    reference: dict[str, Any] = {"expected_answer": example["answer"]}
    if example.get("solution"):
        reference["expert_solution"] = example["solution"]

    return {
        "prompt": [{"role": "user", "content": MATH_INSTRUCTION_PREFIX + example["problem"]}],
        "reference": json.dumps(reference, ensure_ascii=True),
        "data_source": DATA_SOURCE,
        "meta_information": json.dumps(
            {
                "problem_type": example.get("problem_type"),
                "question_type": example.get("question_type"),
                "source": example.get("source"),
            },
            ensure_ascii=True,
        ),
        "data_source_id": str(idx),
        "turn": 0,
    }


map_numina_math = row_mapper_to_batched(_map_numina_math_row)
