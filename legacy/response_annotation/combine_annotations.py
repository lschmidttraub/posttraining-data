"""
Build a combined annotated dataset from per-model annotation results.

For each prompt row, adds a single 'annotations' column containing a JSON array
of objects — one per model — with: model name, detailed annotation scores, and
a final_score (mean of aspect expected scores).

The original row count is preserved (no stacking).
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Value, load_from_disk

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path("/iopsstor/scratch/cscs/dmelikidze/posttraining-data/response_annotation/datasets/inference_results_final")
OUT_DIR = Path("/iopsstor/scratch/cscs/dmelikidze/posttraining-data/response_annotation/datasets/combined_annotated")

ASPECTS = ["helpfulness", "honesty", "instruction_following", "truthfulness"]
NUM_PROC = min(os.cpu_count() or 4, 288)

print(f"Using {NUM_PROC} processes.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def expected_score(dist: dict) -> float:
    return sum(float(k) * float(v) for k, v in dist.items())


def annotation_to_score(annotation: dict) -> float:
    scores = [expected_score(annotation[asp]) for asp in ASPECTS if asp in annotation]
    return sum(scores) / len(scores) if scores else 0.0


def load_model_data(model_dir: Path) -> tuple[str, list[dict], list[float], list[str]]:
    """Load a single model dataset and extract annotations, scores, and responses."""
    name = model_dir.name
    ds = load_from_disk(str(model_dir))
    annotations = ds["annotation"]
    responses = ds["response"]
    scores = [annotation_to_score(ann) for ann in annotations]
    return name, annotations, scores, responses


# ---------------------------------------------------------------------------
# Load all model annotations in parallel using ProcessPoolExecutor
# ---------------------------------------------------------------------------
model_dirs = sorted([p for p in BASE_DIR.iterdir() if p.is_dir()])
print(f"Found {len(model_dirs)} model directories: {[p.name for p in model_dirs]}")

print("Loading and scoring all model datasets in parallel ...", flush=True)

model_data: dict[str, tuple[list[dict], list[float], list[str]]] = {}

with ProcessPoolExecutor(max_workers=min(len(model_dirs), NUM_PROC)) as executor:
    futures = {executor.submit(load_model_data, d): d for d in model_dirs}
    for future in as_completed(futures):
        name, annotations, scores, responses = future.result()
        model_data[name] = (annotations, scores, responses)
        print(f"  Loaded {name} ({len(annotations)} rows)", flush=True)

# Verify all datasets have the same length
lengths = {name: len(anns) for name, (anns, _, _) in model_data.items()}
assert len(set(lengths.values())) == 1, f"Datasets have different lengths: {lengths}"
num_rows = next(iter(lengths.values()))
print(f"\nAll datasets have {num_rows} rows.")

# ---------------------------------------------------------------------------
# Use one model's dataset as the base (it has the shared columns)
# ---------------------------------------------------------------------------
ref_name = model_dirs[0].name
print(f"Loading base dataset from {ref_name} for shared columns ...", flush=True)
base_ds = load_from_disk(str(model_dirs[0]))

# ---------------------------------------------------------------------------
# Build the annotations column: one JSON array per row
# ---------------------------------------------------------------------------
model_names = sorted(model_data.keys())

# Pre-extract into lists for fast indexed access
_all_annotations = {name: model_data[name][0] for name in model_names}
_all_scores = {name: model_data[name][1] for name in model_names}
_all_responses = {name: model_data[name][2] for name in model_names}
_model_names = model_names


def build_annotations_column(batch, indices):
    """For each row, build a JSON array of annotation objects across all models."""
    results = []
    for idx in indices:
        row_annotations = []
        for name in _model_names:
            row_annotations.append({
                "model": name,
                "response": _all_responses[name][idx],
                "detailed_annotations": _all_annotations[name][idx],
                "final_score": round(_all_scores[name][idx], 4),
            })
        results.append(json.dumps(row_annotations, ensure_ascii=False))
    return {"annotations": results}


print("Building annotations column ...", flush=True)
base_ds = base_ds.map(
    build_annotations_column,
    batched=True,
    batch_size=2000,
    with_indices=True,
    num_proc=NUM_PROC,
    desc="Building annotations",
)

# Drop the per-model annotation column (now redundant) and any internal columns
columns_to_remove = [c for c in base_ds.column_names if c in ("annotation", "_score")]
if columns_to_remove:
    base_ds = base_ds.remove_columns(columns_to_remove)

print(f"\nFinal dataset:")
print(base_ds)
print(f"Columns: {base_ds.column_names}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
print(f"\nSaving combined dataset to {OUT_DIR} ...", flush=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
DatasetDict({"train": base_ds}).save_to_disk(str(OUT_DIR))

print("Done.")
