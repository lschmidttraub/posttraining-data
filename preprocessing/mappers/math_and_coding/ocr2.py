import json
from typing import Any, Dict, Optional

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from preprocessing.mappers.utils import row_mapper_to_batched

DATA_SOURCE = "nvidia/OpenCodeReasoning-2"

OCR2_INSTRUCTION_PREFIX = {
    "cpp": (
        "Solve the following programming problem. Explain your approach, "
        "then provide the complete solution in C++.\n\n"
        "Your final code must be in a single ```cpp block:\n"
        "```cpp\n// your code here\n```\n\n"
    ),
    "python": (
        "Solve the following programming problem. Explain your approach, "
        "then provide the complete solution in Python.\n\n"
        "Your final code must be in a single ```python block:\n"
        "```python\n# your code here\n```\n\n"
    ),
}

SELECTION_STRATEGY = "max_pass_rate_then_prefer_right_then_longer_solution"


def _load_ocr2_source_dataset(src_name: str):
    if src_name == "taco":
        return load_dataset(
            "parquet",
            data_files={"train": "hf://datasets/BAAI/TACO/ALL/train-*.parquet"},
            split="train",
        )
    elif src_name == "apps":
        return load_dataset(
            "parquet",
            data_files={
                "train": "hf://datasets/codeparrot/apps@refs/pr/5/all/train-*",
                "test": "hf://datasets/codeparrot/apps@refs/pr/5/all/test-*",
            },
        )
    elif src_name == "code_contests":
        return load_dataset("deepmind/code_contests")
    elif src_name == "open-r1/codeforces":
        return load_dataset("open-r1/codeforces")
    else:
        raise ValueError(f"Unknown source dataset: {src_name}")


def _extract_question_from_source(src_name: str, src_row: Dict[str, Any]) -> str:
    if src_name in ("taco", "apps"):
        return src_row.get("question", "")
    elif src_name == "code_contests":
        return src_row.get("description", "")
    elif src_name == "open-r1/codeforces":
        question = src_row.get("description", "")
        if src_row.get("input_format"):
            question += "\n\nInput\n\n" + src_row["input_format"]
        if src_row.get("output_format"):
            question += "\n\nOutput\n\n" + src_row["output_format"]
        if src_row.get("examples"):
            question += "\n\nExamples"
            for example in src_row["examples"]:
                if "input" in example:
                    question += "\n\nInput\n\n" + example["input"]
                if "output" in example:
                    question += "\n\nOutput\n\n" + example["output"]
        if src_row.get("note"):
            question += "\n\nNote\n\n" + src_row["note"]
        return question
    return ""


def _reconstruct_question(
    example: Dict[str, Any], source_cache: Dict[str, Any]
) -> str:
    src_name = example.get("dataset")
    src_split = example.get("split")
    src_idx = example.get("index")

    if not src_name or not src_split or src_idx is None:
        raise ValueError(
            f"Missing source metadata: dataset={src_name}, split={src_split}, index={src_idx}"
        )

    src_idx = int(src_idx)

    if src_name not in source_cache:
        source_cache[src_name] = _load_ocr2_source_dataset(src_name)

    src_ds = source_cache[src_name]
    if hasattr(src_ds, "keys"):  # DatasetDict
        src_row = src_ds[src_split][src_idx]
    else:
        src_row = src_ds[src_idx]
    question = _extract_question_from_source(src_name, src_row)
    if not question.strip():
        raise ValueError(
            f"Empty question after reconstruction: dataset={src_name}, split={src_split}, index={src_idx}"
        )
    return question


def _select_best_per_problem(ds: Dataset, split: str) -> Dataset:
    """Group by question_id, select best attempt per problem, reconstruct questions."""
    df = ds.to_pandas()
    problem_id_col = "question_id" if "question_id" in df.columns else "id"

    df["pass_rate_num"] = pd.to_numeric(df["pass_rate"], errors="coerce")
    df["is_right"] = df["judgement"].astype(str).str.lower().eq("right")
    df["solution_len"] = df["r1_generation"].fillna("").astype(str).str.len()

    problem_stats = df.groupby(problem_id_col, dropna=False).agg(
        num_attempts=("id", "size"),
        has_right_attempt=("is_right", "max"),
        max_pass_rate=("pass_rate_num", "max"),
    ).reset_index()
    df = df.merge(problem_stats, on=problem_id_col, how="left")

    best_per_problem = (
        df.sort_values(
            by=["pass_rate_num", "is_right", "solution_len"],
            ascending=[False, False, False],
            na_position="last",
        )
        .drop_duplicates(subset=[problem_id_col], keep="first")
    )

    best_ds = Dataset.from_pandas(best_per_problem, preserve_index=False)

    source_cache: Dict[str, Any] = {}
    questions: list[str] = []
    for row in best_ds:
        questions.append(_reconstruct_question(row, source_cache))

    best_ds = best_ds.add_column("_reconstructed_question", questions)
    best_ds = best_ds.add_column("_ocr2_split", [split] * len(best_ds))
    return best_ds


def load_ocr2() -> DatasetDict:
    """Load both OCR2 splits, select best per problem, reconstruct questions."""
    splits = []
    for split_name in ("cpp", "python"):
        ds = load_dataset(DATA_SOURCE, split=split_name)
        splits.append(_select_best_per_problem(ds, split_name))
    combined = concatenate_datasets(splits)
    return DatasetDict({"train": combined})


def _map_ocr2_row(example: dict[str, Any], idx: int) -> Optional[dict[str, Any]]:
    split = example.get("_ocr2_split", "cpp")
    question = example.get("_reconstructed_question", "")
    prompt_text = OCR2_INSTRUCTION_PREFIX[split] + question

    reference: dict[str, Any] = {}
    if example.get("r1_generation"):
        reference["candidate_solution"] = example["r1_generation"]
    pass_rate = example.get("pass_rate_num")
    if pass_rate is not None and pd.notna(pass_rate):
        reference["pass_rate"] = float(pass_rate)
    if example.get("judgement") is not None:
        reference["judgement"] = str(example["judgement"])

    meta_information = {
        "original_dataset": example.get("dataset"),
        "original_split": example.get("split"),
        "original_index": int(example["index"]) if example.get("index") is not None else None,
        "question_id": example.get("question_id"),
        "difficulty": example.get("difficulty"),
        "source": example.get("source"),
        "selection_strategy": SELECTION_STRATEGY,
        "num_attempts": int(example["num_attempts"]),
        "has_right_attempt": bool(example["has_right_attempt"]),
        "max_pass_rate": float(example["max_pass_rate"]) if pd.notna(example.get("max_pass_rate")) else None,
    }

    return {
        "prompt": [{"role": "user", "content": prompt_text}],
        "reference": json.dumps(reference, ensure_ascii=True),
        "data_source": DATA_SOURCE,
        "meta_information": json.dumps(meta_information, ensure_ascii=True),
        "data_source_id": str(example["id"]),
        "turn": 0,
    }


map_ocr2 = row_mapper_to_batched(_map_ocr2_row)
