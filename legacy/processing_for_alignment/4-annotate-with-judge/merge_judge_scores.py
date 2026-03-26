"""
Stage 4c: Merge judge annotations back into the original dataset.

Takes the annotated exploded dataset (output of annotate.py) and groups
annotations back by original row, producing a list of annotation dicts
per prompt (one per reference completion).

Input:  - Original dataset with reference_completions (output of stage 2)
        - Annotated exploded dataset (output of annotate.py on stage 4a output)
Output: Original dataset with added 'reference_annotations' column
        (list of annotation dicts, one per completion).

Usage:
    python merge_judge_scores.py \
        --original-dataset-path /path/to/dataset_with_completions \
        --annotated-dataset-path /path/to/annotated_exploded_dataset \
        --output-dir /path/to/output
"""

import argparse
import math
from datasets import load_from_disk


def compute_expected_score(annotation_dict):
    """Compute weighted average score from probability distribution over ratings."""
    scores = {}
    for aspect, probs in annotation_dict.items():
        expected = sum(int(k) * v for k, v in probs.items())
        scores[aspect] = expected
    scores["mean"] = sum(scores.values()) / len(scores) if scores else 0.0
    return scores


def main(args):
    print(f"Loading original dataset from {args.original_dataset_path}")
    original = load_from_disk(args.original_dataset_path)

    print(f"Loading annotated dataset from {args.annotated_dataset_path}")
    annotated = load_from_disk(args.annotated_dataset_path)

    n_original = len(original)
    n_completions = len(original[0]["reference_completions"])

    # Group annotations by original_index
    annotations_by_row = {i: [None] * n_completions for i in range(n_original)}

    for row in annotated:
        oi = row["original_index"]
        ci = row["completion_index"]
        annotation = row.get("annotation", {})
        scores = compute_expected_score(annotation) if annotation else {}
        annotations_by_row[oi][ci] = {
            "annotation": annotation,
            "scores": scores,
        }

    # Build column
    reference_annotations = [annotations_by_row[i] for i in range(n_original)]

    original = original.add_column("reference_annotations", reference_annotations)

    print(f"Saving to {args.output_dir}")
    original.save_to_disk(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge judge annotations back into original dataset")
    parser.add_argument("--original-dataset-path", type=str, required=True, help="Path to dataset with reference_completions")
    parser.add_argument("--annotated-dataset-path", type=str, required=True, help="Path to annotated exploded dataset (output of annotate.py)")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save merged dataset")
    args = parser.parse_args()
    main(args)
