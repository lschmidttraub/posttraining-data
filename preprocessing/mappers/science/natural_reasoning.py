import json
from typing import Any

from preprocessing.mappers.utils import inject_system_prompt
from preprocessing.system_prompts import SYSTEM_PROMPT_SCIENCE


DATA_SOURCE = "facebook/natural_reasoning"


def map_natural_reasoning(
    batch: dict[str, list[Any]], indices: list[int]
) -> dict[str, list[Any]]:
    prompts = []
    references = []
    data_sources = []
    data_source_ids = []
    meta_information = []
    turns = []

    questions = batch.get("question", [])
    reference_answers = batch.get("reference_answer", [])
    responses_batch = batch.get("responses", [])

    for offset, row_idx in enumerate(indices):
        question = questions[offset] if offset < len(questions) else ""
        reference_answer = reference_answers[offset] if offset < len(reference_answers) else ""
        responses = responses_batch[offset] if offset < len(responses_batch) else []

        ref = {"reference_answer": reference_answer} if reference_answer else {}

        prompts.append(
            inject_system_prompt(
                [{"role": "user", "content": question}],
                SYSTEM_PROMPT_SCIENCE,
            )
        )
        references.append(json.dumps(ref, ensure_ascii=True))
        data_sources.append(DATA_SOURCE)
        data_source_ids.append(str(row_idx))
        meta_information.append(
            json.dumps({"responses": responses}, ensure_ascii=True)
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
