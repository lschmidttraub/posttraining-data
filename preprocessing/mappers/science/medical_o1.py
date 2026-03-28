import json
from typing import Any


DATA_SOURCE = "FreedomIntelligence/medical-o1-verifiable-problem"

MEDICAL_O1_VERIFIER_PROMPT_TEMPLATE = '''<Model Response>
{response}
</Model Response>

<Reference Answer>
{reference}
</Reference Answer>

Your task is to evaluate the model response by comparing it to the reference answer. If the model response is correct and aligns with the reference answer, output "True" . If it is incorrect or fails to select the correct option (if options are provided), output "False"'''


def map_medical_o1(
    batch: dict[str, list[Any]], indices: list[int]
) -> dict[str, list[Any]]:
    prompts = []
    references = []
    data_sources = []
    data_source_ids = []
    meta_information = []
    turns = []

    questions = batch.get("Open-ended Verifiable Question", [])
    ground_truth_answers = batch.get("Ground-True Answer", [])

    for offset, row_idx in enumerate(indices):
        question = questions[offset] if offset < len(questions) else ""
        ground_truth = ground_truth_answers[offset] if offset < len(ground_truth_answers) else ""

        prompts.append([{"role": "user", "content": question}])
        references.append(
            json.dumps(
                {
                    "ground_truth_answer": ground_truth,
                    "verifier_model_path": "FreedomIntelligence/medical_o1_verifier_3B",
                    "verifier_prompt_template": MEDICAL_O1_VERIFIER_PROMPT_TEMPLATE,
                },
                ensure_ascii=True,
            )
        )
        data_sources.append(DATA_SOURCE)
        data_source_ids.append(str(row_idx))
        meta_information.append(json.dumps({}, ensure_ascii=True))
        turns.append(0)

    return {
        "prompt": prompts,
        "reference": references,
        "data_source": data_sources,
        "data_source_id": data_source_ids,
        "meta_information": meta_information,
        "turn": turns,
    }
