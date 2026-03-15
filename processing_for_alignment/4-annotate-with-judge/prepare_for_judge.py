"""
Stage 4a: Prepare reference completions for LLM-as-a-Judge annotation.

Takes the dataset with reference_completions (output of stage 2) and
"explodes" it into individual rows — one per (prompt, completion) pair —
in the format expected by response_annotation/annotate.py.

Input:  HuggingFace dataset with 'chosen' and 'reference_completions' columns.
Output: HuggingFace dataset with:
        - 'chosen': prompt messages (without final assistant turn)
        - 'response': single completion string
        - 'original_index': index in the source dataset
        - 'completion_index': which of the N completions this is

Usage:
    python prepare_for_judge.py \
        --dataset-path /path/to/dataset_with_completions \
        --output-dir /path/to/exploded_dataset
"""

import argparse
from datasets import Dataset, load_from_disk


def main(args):
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    print(f"Dataset size: {len(dataset)}")

    rows = []
    for i, example in enumerate(dataset):
        # Prompt = chosen without last assistant message
        prompt_messages = example["chosen"]
        if prompt_messages[-1]["role"] == "assistant":
            prompt_messages = prompt_messages[:-1]

        for j, completion in enumerate(example["reference_completions"]):
            rows.append({
                "chosen": prompt_messages,
                "response": completion,
                "original_index": i,
                "completion_index": j,
            })

    exploded = Dataset.from_list(rows)
    n_completions = len(dataset[0]["reference_completions"])
    print(f"Exploded: {len(dataset)} rows x {n_completions} completions = {len(exploded)} rows")

    print(f"Saving to {args.output_dir}")
    exploded.save_to_disk(args.output_dir)
    print("Done. Run annotate.py on this dataset with --prompt-column-name chosen --remove-last-message")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare reference completions for judge annotation")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to dataset with reference_completions")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save exploded dataset")
    args = parser.parse_args()
    main(args)
