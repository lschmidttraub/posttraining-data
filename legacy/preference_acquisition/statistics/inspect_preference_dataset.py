"""
Inspect a preference dataset: tokenize chosen/rejected, report token length
statistics, and show how many rows would be filtered at various thresholds.
"""

import os

import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_PATH = "/iopsstor/scratch/cscs/dmelikidze/posttraining-data/preference_acquisition/datasets/MaxMin"
MODEL_NAME_OR_PATH = "/iopsstor/scratch/cscs/dmelikidze/aper_mods/apertus1-base-sft-stage1"
NUM_PROC = min(os.cpu_count() or 4, 288)

print(f"Using {NUM_PROC} processes.")
print(f"Tokenizer: {MODEL_NAME_OR_PATH}")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)

dataset = load_from_disk(DATASET_PATH)
if "train_split" in dataset:
    dataset = dataset["train_split"]
elif "train" in dataset:
    dataset = dataset["train"]

print(f"Loaded dataset: {len(dataset)} rows, columns: {dataset.column_names}\n")


# ---------------------------------------------------------------------------
# Tokenize chosen and rejected
# ---------------------------------------------------------------------------
def sanitize_messages(messages):
    """Ensure all message content values are strings."""
    return [{"role": str(m["role"]), "content": str(m.get("content", ""))} for m in messages]


def compute_token_lengths(batch):
    chosen_lens = []
    rejected_lens = []
    for chosen, rejected in zip(batch["chosen"], batch["rejected"]):
        chosen_len = len(tokenizer.apply_chat_template(sanitize_messages(chosen), tokenize=True))
        rejected_len = len(tokenizer.apply_chat_template(sanitize_messages(rejected), tokenize=True))
        chosen_lens.append(chosen_len)
        rejected_lens.append(rejected_len)
    return {"chosen_tokens": chosen_lens, "rejected_tokens": rejected_lens}


print("Tokenizing chosen and rejected ...", flush=True)
dataset = dataset.map(
    compute_token_lengths,
    batched=True,
    batch_size=2000,
    num_proc=NUM_PROC,
    desc="Counting tokens",
)

chosen_tokens = np.array(dataset["chosen_tokens"])
rejected_tokens = np.array(dataset["rejected_tokens"])
max_tokens = np.maximum(chosen_tokens, rejected_tokens)

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
print("\n=== Token Length Statistics ===")
for name, arr in [("Chosen", chosen_tokens), ("Rejected", rejected_tokens), ("Max(chosen, rejected)", max_tokens)]:
    print(f"\n{name}:")
    print(f"  Mean:   {arr.mean():.1f}")
    print(f"  Median: {np.median(arr):.1f}")
    print(f"  Std:    {arr.std():.1f}")
    print(f"  Min:    {arr.min()}")
    print(f"  Max:    {arr.max()}")
    for p in [90, 95, 99]:
        print(f"  P{p}:    {np.percentile(arr, p):.0f}")

# ---------------------------------------------------------------------------
# Filtering analysis at various thresholds
# ---------------------------------------------------------------------------
print("\n=== Rows filtered at various max_tokens thresholds ===")
print(f"{'Threshold':>10} {'Chosen>':>10} {'Rejected>':>10} {'Either>':>10} {'Both>':>10} {'Remaining':>10} {'% Kept':>8}")
print("-" * 68)

total = len(dataset)
for threshold in [1024, 2048, 3072, 4096, 6144, 8192]:
    chosen_over = int((chosen_tokens > threshold).sum())
    rejected_over = int((rejected_tokens > threshold).sum())
    either_over = int(((chosen_tokens > threshold) | (rejected_tokens > threshold)).sum())
    both_over = int(((chosen_tokens > threshold) & (rejected_tokens > threshold)).sum())
    remaining = total - either_over
    pct = remaining / total * 100
    print(f"{threshold:>10} {chosen_over:>10} {rejected_over:>10} {either_over:>10} {both_over:>10} {remaining:>10} {pct:>7.1f}%")

# ---------------------------------------------------------------------------
# Model distribution
# ---------------------------------------------------------------------------
print("\n=== Chosen/Rejected Model Distribution ===")
from collections import Counter

chosen_models = Counter(dataset["chosen_model"])
rejected_models = Counter(dataset["rejected_model"])

print(f"\n{'Model':<45} {'Chosen':>8} {'Rejected':>8}")
print("-" * 65)
all_models = sorted(set(chosen_models) | set(rejected_models))
for model in all_models:
    print(f"{model:<45} {chosen_models.get(model, 0):>8} {rejected_models.get(model, 0):>8}")

print(f"\nTotal rows: {total}")
