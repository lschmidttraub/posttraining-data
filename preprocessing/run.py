import argparse
import json
import os
from collections import defaultdict

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk

from preprocessing.registry import MAPPERS
from preprocessing.schema import STANDARD_COLUMNS


def load_input_dataset(dataset_name: str, split: str | None) -> DatasetDict:
    if os.path.exists(dataset_name):
        dataset = load_from_disk(dataset_name)
    else:
        if split is None:
            dataset = load_dataset(dataset_name)
        else:
            dataset = DatasetDict({split: load_dataset(dataset_name, split=split)})

    if isinstance(dataset, DatasetDict):
        return dataset

    active_split = split or "train"
    return DatasetDict({active_split: dataset})


def normalize_prompt(prompt: object) -> str:
    if isinstance(prompt, str):
        return prompt

    try:
        return json.dumps(prompt, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return str(prompt)


def deduplicate_by_prompt(dataset: Dataset, desc: str) -> Dataset:
    seen_prompts: set[str] = set()

    def keep_first_prompt(sample: dict[str, object]) -> bool:
        prompt_key = normalize_prompt(sample.get("prompt"))
        if prompt_key in seen_prompts:
            return False
        seen_prompts.add(prompt_key)
        return True

    original_size = len(dataset)
    deduplicated = dataset.filter(keep_first_prompt, desc=desc)
    removed = original_size - len(deduplicated)
    print(f"{desc}: removed {removed} duplicate prompts, kept {len(deduplicated)} rows")
    return deduplicated


def validate_parallel_args(
    datasets: list[str], mappers: list[str], splits: list[str | None]
) -> list[tuple[str, str, str | None]]:
    if len(mappers) not in (1, len(datasets)):
        raise ValueError("Provide either one --mapper for all datasets or one --mapper per --dataset")

    if len(splits) not in (0, 1, len(datasets)):
        raise ValueError("Provide either one --split for all datasets or one --split per --dataset")

    resolved_mappers = mappers if len(mappers) == len(datasets) else mappers * len(datasets)
    resolved_splits = splits if len(splits) == len(datasets) else (splits * len(datasets) if splits else [None] * len(datasets))
    return list(zip(datasets, resolved_mappers, resolved_splits))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map one or more datasets into the standardized preprocessing schema and combine them."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="HuggingFace dataset name or local dataset path. Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--mapper",
        action="append",
        required=True,
        choices=sorted(MAPPERS),
        help="Mapper function to apply. Repeat to align with multiple --dataset values.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory where the processed dataset will be saved")
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        help="Optional split name when loading from the Hub. Repeat to align with multiple --dataset values.",
    )
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for dataset.map")
    parser.add_argument("--num-proc", type=int, default=None, help="Optional number of worker processes for dataset.map")
    parser.add_argument("--upload-to-hub", action="store_true", help="Upload the processed dataset to the Hugging Face Hub")
    parser.add_argument("--hub-dataset-id", default=None, help="Target Hugging Face dataset repo, e.g. org/name")
    parser.add_argument("--hub-private", action="store_true", help="Create or update the Hub dataset as private")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_specs = validate_parallel_args(args.dataset, args.mapper, args.split)

    processed_splits: dict[str, list[Dataset]] = defaultdict(list)
    for dataset_name, mapper_name, split in dataset_specs:
        mapper = MAPPERS[mapper_name]
        dataset = load_input_dataset(dataset_name, split)

        for split_name, split_dataset in dataset.items():
            processed_split = split_dataset.map(
                mapper,
                batched=True,
                with_indices=True,
                batch_size=args.batch_size,
                num_proc=args.num_proc,
                remove_columns=split_dataset.column_names,
                desc=f"Preprocessing {dataset_name}:{split_name}",
            ).select_columns(STANDARD_COLUMNS)
            processed_splits[split_name].append(processed_split)

    processed = DatasetDict()
    for split_name, split_datasets in processed_splits.items():
        combined_split = split_datasets[0] if len(split_datasets) == 1 else concatenate_datasets(split_datasets)
        processed[split_name] = deduplicate_by_prompt(
            combined_split,
            desc=f"Deduplicating prompt rows in split '{split_name}'",
        )

    os.makedirs(args.output_dir, exist_ok=True)
    processed.save_to_disk(args.output_dir)
    print(f"Saved preprocessed dataset to {args.output_dir}")

    if args.upload_to_hub:
        if not args.hub_dataset_id:
            raise ValueError("--hub-dataset-id is required when --upload-to-hub is set")
        processed.push_to_hub(args.hub_dataset_id, private=args.hub_private)
        print(f"Uploaded preprocessed dataset to {args.hub_dataset_id}")


if __name__ == "__main__":
    main()
