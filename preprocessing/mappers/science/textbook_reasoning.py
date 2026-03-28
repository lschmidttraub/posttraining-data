import json
from typing import Any


DATA_SOURCE = "MegaScience/TextbookReasoning"


def map_textbook_reasoning(
    batch: dict[str, list[Any]], indices: list[int]
) -> dict[str, list[Any]]:
    prompts = []
    references = []
    data_sources = []
    data_source_ids = []
    meta_information = []
    turns = []

    questions = batch.get("question", [])
    answers = batch.get("answer", [])
    subjects = batch.get("subject", [])
    reference_answers = batch.get("reference_answer", [])

    for offset, row_idx in enumerate(indices):
        question = questions[offset] if offset < len(questions) else ""
        answer = answers[offset] if offset < len(answers) else ""
        subject = subjects[offset] if offset < len(subjects) else ""
        reference_answer = reference_answers[offset] if offset < len(reference_answers) else ""

        prompts.append([{"role": "user", "content": question}])
        references.append(json.dumps({"reference_answer": reference_answer}, ensure_ascii=True))
        data_sources.append(DATA_SOURCE)
        data_source_ids.append(str(row_idx))
        meta_information.append(
            json.dumps(
                {
                    "subject": subject,
                    "reasoning_answer": answer,
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
