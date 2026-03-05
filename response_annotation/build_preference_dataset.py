"""
Build a preference dataset from per-model annotation datasets.
"""

import json
import os
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_from_disk

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path("/iopsstor/scratch/cscs/dmelikidze/posttraining-data/response_annotation/datasets/inference_results_final")
OUT_PAIRWISE_DIR = Path("/iopsstor/scratch/cscs/dmelikidze/posttraining-data/response_generation/datasets/preference/maxmin")
OUT_ALL_DIR = Path("/iopsstor/scratch/cscs/dmelikidze/posttraining-data/response_annotation/datasets/all_completions_annotated")

ASPECTS = ["helpfulness", "honesty", "instruction_following", "truthfulness"]
NUM_PROC = min(os.cpu_count(), 128)

print(f"Using {NUM_PROC} processes.")

# ---------------------------------------------------------------------------
# Explicit feature schemas
# Sequence({"role": ..., "content": ...}) in HuggingFace Datasets is stored
# as struct-of-lists (columnar). To hold list-of-structs, use Value("string")
# with JSON serialisation — this is the most reliable cross-version approach.
# ---------------------------------------------------------------------------
PAIRWISE_FEATURES = Features({
    "chosen":         Value("string"),  # JSON list of {"role":..,"content":..}
    "rejected":       Value("string"),  # JSON list of {"role":..,"content":..}
    "prompt_id":      Value("string"),
    "chosen_model":   Value("string"),
    "rejected_model": Value("string"),
    "chosen_score":   Value("float32"),
    "rejected_score": Value("float32"),
})

ALL_FEATURES = Features({
    "prompt":    Value("string"),       # JSON list of {"role":..,"content":..}
    "response":  Value("string"),
    "model":     Value("string"),
    "prompt_id": Value("string"),
    "score":     Value("float32"),
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def expected_score(dist: dict) -> float:
    return sum(float(k) * float(v) for k, v in dist.items())

def annotation_to_score(annotation: dict) -> float:
    scores = [expected_score(annotation[asp]) for asp in ASPECTS if asp in annotation]
    return sum(scores) / len(scores) if scores else 0.0

def add_score_column(batch):
    return {"_score": [annotation_to_score(ann) for ann in batch["annotation"]]}

def strip_message(msg: dict) -> dict:
    """Keep only role and content from a conversation message."""
    return {"role": str(msg["role"]), "content": str(msg["content"])}

def msgs_to_json(messages: list) -> str:
    return json.dumps([strip_message(m) for m in messages], ensure_ascii=False)

# ---------------------------------------------------------------------------
# Load all model datasets and compute scores in parallel
# ---------------------------------------------------------------------------
model_dirs = sorted([p for p in BASE_DIR.iterdir() if p.is_dir()])
model_names = [p.name for p in model_dirs]
print(f"Found {len(model_names)} model datasets: {model_names}")

datasets = {}
scores_per_model = {}

for model_dir in model_dirs:
    name = model_dir.name
    print(f"Loading {name} ...", flush=True)
    ds = load_from_disk(str(model_dir))

    print(f"  Scoring {name} ...", flush=True)
    ds = ds.map(
        add_score_column,
        batched=True,
        batch_size=2000,
        num_proc=NUM_PROC,
        desc=f"Scoring {name}",
    )
    scores_per_model[name] = np.array(ds["_score"], dtype=np.float32)
    datasets[name] = ds

lengths = {n: len(d) for n, d in datasets.items()}
assert len(set(lengths.values())) == 1, f"Datasets have different lengths: {lengths}"
num_rows = next(iter(lengths.values()))
print(f"\nAll datasets have {num_rows} rows.")

# ---------------------------------------------------------------------------
# Vectorised best/worst selection
# ---------------------------------------------------------------------------
print("Selecting best/worst models per row ...", flush=True)

score_matrix = np.stack([scores_per_model[n] for n in model_names], axis=1)

best_idx  = score_matrix.argmax(axis=1)
worst_idx = score_matrix.argmin(axis=1)

best_score  = score_matrix[np.arange(num_rows), best_idx]
worst_score = score_matrix[np.arange(num_rows), worst_idx]

tied_mask     = best_idx == worst_idx
valid_indices = np.where(~tied_mask)[0]
print(f"  Valid rows (not tied): {len(valid_indices)} / {num_rows}")

# ---------------------------------------------------------------------------
# Pre-extract columns
# ---------------------------------------------------------------------------
print("Extracting columns ...", flush=True)

ref_ds        = datasets[model_names[0]]
chosen_col    = ref_ds["chosen"]
prompt_id_col = ref_ds["prompt_id"]
responses     = {name: datasets[name]["response"] for name in model_names}

_chosen_col    = chosen_col
_prompt_id_col = prompt_id_col
_responses     = responses
_model_names   = model_names

# ---------------------------------------------------------------------------
# Build pairwise dataset
# ---------------------------------------------------------------------------
print("Building pairwise index dataset ...", flush=True)

idx_ds = Dataset.from_dict({
    "row_idx":         valid_indices.tolist(),
    "best_model_idx":  best_idx[valid_indices].tolist(),
    "worst_model_idx": worst_idx[valid_indices].tolist(),
    "chosen_score":    best_score[valid_indices].tolist(),
    "rejected_score":  worst_score[valid_indices].tolist(),
})

def expand_pairwise(batch):
    chosen_msgs     = []
    rejected_msgs   = []
    prompt_ids      = []
    chosen_models   = []
    rejected_models = []
    chosen_scores   = []
    rejected_scores = []

    for row_idx, b_idx, w_idx, c_score, r_score in zip(
        batch["row_idx"],
        batch["best_model_idx"],
        batch["worst_model_idx"],
        batch["chosen_score"],
        batch["rejected_score"],
    ):
        prompt_messages = [strip_message(m) for m in _chosen_col[row_idx][:-1]]
        b_name = _model_names[b_idx]
        w_name = _model_names[w_idx]

        chosen_msgs.append(json.dumps(
            prompt_messages + [{"role": "assistant", "content": str(_responses[b_name][row_idx])}],
            ensure_ascii=False
        ))
        rejected_msgs.append(json.dumps(
            prompt_messages + [{"role": "assistant", "content": str(_responses[w_name][row_idx])}],
            ensure_ascii=False
        ))
        prompt_ids.append(str(_prompt_id_col[row_idx]))
        chosen_models.append(b_name)
        rejected_models.append(w_name)
        chosen_scores.append(float(c_score))
        rejected_scores.append(float(r_score))

    return {
        "chosen":         chosen_msgs,
        "rejected":       rejected_msgs,
        "prompt_id":      prompt_ids,
        "chosen_model":   chosen_models,
        "rejected_model": rejected_models,
        "chosen_score":   chosen_scores,
        "rejected_score": rejected_scores,
    }

print("Expanding pairwise rows ...", flush=True)
pairwise_ds = idx_ds.map(
    expand_pairwise,
    batched=True,
    batch_size=2000,
    num_proc=NUM_PROC,
    remove_columns=idx_ds.column_names,
    features=PAIRWISE_FEATURES,
    desc="Expanding pairwise",
)

print(pairwise_ds)

# ---------------------------------------------------------------------------
# Save pairwise dataset FIRST
# ---------------------------------------------------------------------------
print(f"\nSaving pairwise dataset to {OUT_PAIRWISE_DIR} ...", flush=True)
OUT_PAIRWISE_DIR.mkdir(parents=True, exist_ok=True)
DatasetDict({"train": pairwise_ds}).save_to_disk(str(OUT_PAIRWISE_DIR))
print("Pairwise dataset saved successfully.", flush=True)

# ---------------------------------------------------------------------------
# Build all-completions dataset
# ---------------------------------------------------------------------------
print("Building all-completions index dataset ...", flush=True)

all_row_idxs   = np.repeat(valid_indices, len(model_names))
all_model_idxs = np.tile(np.arange(len(model_names)), len(valid_indices))

all_idx_ds = Dataset.from_dict({
    "row_idx":   all_row_idxs.tolist(),
    "model_idx": all_model_idxs.tolist(),
    "score":     score_matrix[all_row_idxs, all_model_idxs].tolist(),
})

def expand_all(batch):
    prompts       = []
    responses_out = []
    models_out    = []
    prompt_ids    = []
    scores_out    = []

    for row_idx, m_idx, score in zip(
        batch["row_idx"], batch["model_idx"], batch["score"]
    ):
        name = _model_names[m_idx]
        prompts.append(json.dumps(
            [strip_message(m) for m in _chosen_col[row_idx][:-1]],
            ensure_ascii=False
        ))
        responses_out.append(str(_responses[name][row_idx]))
        models_out.append(name)
        prompt_ids.append(str(_prompt_id_col[row_idx]))
        scores_out.append(float(score))

    return {
        "prompt":    prompts,
        "response":  responses_out,
        "model":     models_out,
        "prompt_id": prompt_ids,
        "score":     scores_out,
    }

print("Expanding all-completions rows ...", flush=True)
all_ds = all_idx_ds.map(
    expand_all,
    batched=True,
    batch_size=2000,
    num_proc=NUM_PROC,
    remove_columns=["row_idx", "model_idx"],
    features=ALL_FEATURES,
    desc="Expanding all-completions",
)

print(all_ds)

print(f"Saving all-completions dataset to {OUT_ALL_DIR} ...", flush=True)
OUT_ALL_DIR.mkdir(parents=True, exist_ok=True)
DatasetDict({"train": all_ds}).save_to_disk(str(OUT_ALL_DIR))

print("\nDone.")
