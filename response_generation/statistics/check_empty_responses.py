"""
Check for empty or None responses in each model's generated dataset.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from datasets import load_from_disk

BASE_DIR = Path("/iopsstor/scratch/cscs/dmelikidze/posttraining-data/response_generation/datasets/inference_results_final")
NUM_PROC = min(os.cpu_count() or 4, 288)


def check_model(model_dir: Path) -> tuple[str, int, int, int]:
    """Returns (model_name, total_rows, empty_count, none_count)."""
    name = model_dir.name
    ds = load_from_disk(str(model_dir))
    responses = ds["response"]
    total = len(responses)
    none_count = 0
    empty_count = 0
    for r in responses:
        if r is None:
            none_count += 1
        elif str(r).strip() == "":
            empty_count += 1
    return name, total, empty_count, none_count


model_dirs = sorted([p for p in BASE_DIR.iterdir() if p.is_dir()])
print(f"Found {len(model_dirs)} model directories in {BASE_DIR}")
print(f"Using {NUM_PROC} workers.\n")

results = []
with ProcessPoolExecutor(max_workers=min(len(model_dirs), NUM_PROC)) as executor:
    futures = {executor.submit(check_model, d): d for d in model_dirs}
    for future in as_completed(futures):
        name, total, empty, none = future.result()
        results.append((name, total, empty, none))
        print(f"  Checked {name}: {empty} empty, {none} None (out of {total})", flush=True)

results.sort(key=lambda x: x[0])

print(f"\n{'Model':<45} {'Total':>8} {'Empty':>8} {'None':>8} {'Bad %':>8}")
print("-" * 80)
total_all = 0
total_empty = 0
total_none = 0
for name, total, empty, none in results:
    bad_pct = (empty + none) / total * 100 if total > 0 else 0
    print(f"{name:<45} {total:>8} {empty:>8} {none:>8} {bad_pct:>7.2f}%")
    total_all += total
    total_empty += empty
    total_none += none

print("-" * 80)
bad_pct = (total_empty + total_none) / total_all * 100 if total_all > 0 else 0
print(f"{'TOTAL':<45} {total_all:>8} {total_empty:>8} {total_none:>8} {bad_pct:>7.2f}%")
