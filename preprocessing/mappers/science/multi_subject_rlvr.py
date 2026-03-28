import json
from typing import Any


DATA_SOURCE = "virtuoussy/Multi-subject-RLVR"

MULTI_SUBJECT_RLVR_VERIFIER_PROMPT_TEMPLATE = """
Given a problem, determine whether the final answer in the provided (incomplete) solution process matches the reference answer.
The reference answer may be one single option character (e.g., A, B, C, D), a numerical value, an expression, or a list of answers if multiple questions are involved.
**The reference answer may be in Chinese or another language, but your evaluation should be language-agnostic.**

Your task:
- Compare the final output of the solution process with the reference answer.
- If they **match exactly**, output **YES**.
- If they **do not match**, output **NO**.
- If the solution process is unclear, incomplete, or ambiguous, assume it is incorrect and output **NO**.

Your output must be strictly **'YES'** or **'NO'**, with no additional words, punctuation, or explanation.

---

**Question:**
{question}

**Solution Process (Final Step Only):**
{response}

**Reference Answer:**
{reference}

**Output:**
"""


def _normalize_message(message: dict[str, Any]) -> dict[str, str]:
    return {
        "role": str(message["role"]),
        "content": str(message["content"]),
    }


def map_multi_subject_rlvr(
    batch: dict[str, list[Any]], indices: list[int]
) -> dict[str, list[Any]]:
    prompts = []
    references = []
    data_sources = []
    data_source_ids = []
    meta_information = []
    turns = []

    queries = batch.get("query", [])
    labels = batch.get("label", [])

    for offset, row_idx in enumerate(indices):
        query = queries[offset] if offset < len(queries) else []
        label = labels[offset] if offset < len(labels) else ""

        if len(query) != 2:
            continue

        system_message, user_message = query
        if system_message.get("role") != "system" or user_message.get("role") != "user":
            continue

        prompts.append([
            _normalize_message(system_message),
            _normalize_message(user_message),
        ])
        references.append(
            json.dumps(
                {
                    "ground_truth_answer": label,
                    "verifier_model_path": "virtuoussy/Qwen2.5-7B-Instruct-RLVR",
                    "verifier_prompt_template": MULTI_SUBJECT_RLVR_VERIFIER_PROMPT_TEMPLATE,
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
