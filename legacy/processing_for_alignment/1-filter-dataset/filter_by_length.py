"""
Stage 1: Filter a preference dataset by maximum sequence length.

Tokenizes both 'chosen' and 'rejected' columns and removes rows
where either exceeds max_seq_len tokens.

Input:  HuggingFace dataset with 'chosen' and 'rejected' columns
        (list of message dicts: [{role, content}, ...])
Output: Filtered dataset saved to disk (same schema, fewer rows).

Usage:
    python filter_by_length.py \
        --dataset-path /path/to/dataset \
        --output-dir /path/to/output \
        --model-name-or-path /path/to/model \
        --max-seq-len 4096
"""

import argparse
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer


def count_tokens(messages, tokenizer):
    """Tokenize a conversation and return the token count."""
    token_ids = tokenizer.apply_chat_template(messages, tokenize=True)
    return len(token_ids)


def main(args):
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

    # If loaded as a DatasetDict, extract the train_split
    if hasattr(dataset, "keys"):
        split = "train_split" if "train_split" in dataset else list(dataset.keys())[0]
        print(f"DatasetDict detected, using '{split}' split")
        dataset = dataset[split]

    print(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    original_size = len(dataset)
    print(f"Original dataset size: {original_size}")
    print(f"Filtering with max_seq_len={args.max_seq_len}")

    def is_within_length(example):
        chosen_len = count_tokens(example["chosen"], tokenizer)
        rejected_len = count_tokens(example["rejected"], tokenizer)
        return chosen_len <= args.max_seq_len and rejected_len <= args.max_seq_len

    dataset = dataset.filter(
        is_within_length,
        num_proc=args.num_proc,
        desc="Filtering by sequence length",
    )

    filtered_size = len(dataset)
    removed = original_size - filtered_size
    print(f"Filtered dataset size: {filtered_size} (removed {removed}, {removed/original_size*100:.1f}%)")

    print(f"Saving to {args.output_dir}")
    DatasetDict({"train_split": dataset}).save_to_disk(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter preference dataset by max sequence length")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to input HF dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save filtered dataset")
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Model/tokenizer to use for token counting")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Maximum sequence length in tokens")
    parser.add_argument("--num-proc", type=int, default=64, help="Number of parallel workers for filtering")
    args = parser.parse_args()
    main(args)
