import argparse
import json
import os

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk

from preprocessing.registry import MAPPER_REGISTRY
from preprocessing.schema import STANDARD_COLUMNS


def load_input_dataset(dataset_name: str) -> DatasetDict:
    dataset = load_from_disk(dataset_name) if os.path.exists(dataset_name) else load_dataset(dataset_name)
    if isinstance(dataset, DatasetDict):
        return dataset
    return DatasetDict({"train": dataset})


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


def resolve_category_specs(category: str, datasets: list[str]) -> list[tuple[str, str]]:
    category_mappers = MAPPER_REGISTRY[category]
    if not category_mappers:
        raise ValueError(f"Category '{category}' has no registered mappers")

    if datasets:
        unknown = [d for d in datasets if d not in category_mappers]
        if unknown:
            raise ValueError(
                f"Dataset(s) not registered under category '{category}': {unknown}. "
                f"Available: {sorted(category_mappers)}"
            )
        selected = datasets
    else:
        selected = list(category_mappers.keys())

    return [(dataset, dataset) for dataset in selected]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map one or more datasets into the standardized preprocessing schema and combine them."
    )
    parser.add_argument(
        "--category",
        choices=sorted(MAPPER_REGISTRY),
        required=True,
        help="Category key from MAPPER_REGISTRY. All datasets in the category are processed by default.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=(
            "Restrict processing to this dataset name (must be registered in the category). "
            "Repeat to select multiple. Defaults to all datasets in the category."
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Directory where the processed dataset will be saved")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for dataset.map")
    parser.add_argument("--num-proc", type=int, default=None, help="Optional number of worker processes for dataset.map")
    parser.add_argument("--upload-to-hub", action="store_true", help="Upload the processed dataset to the Hugging Face Hub")
    parser.add_argument("--hub-dataset-id", default=None, help="Target Hugging Face dataset repo, e.g. org/name")
    parser.add_argument("--hub-private", action="store_true", help="Create or update the Hub dataset as private")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_specs = resolve_category_specs(args.category, args.dataset)

    category_mappers = MAPPER_REGISTRY[args.category]

    all_processed: list[Dataset] = []
    for dataset_name, mapper_name in dataset_specs:
        mapper = category_mappers[mapper_name]
        dataset = load_input_dataset(dataset_name)

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
            all_processed.append(processed_split)

    combined = all_processed[0] if len(all_processed) == 1 else concatenate_datasets(all_processed)
    train_split = deduplicate_by_prompt(combined, desc="Deduplicating prompt rows")
    processed = DatasetDict({"train": train_split})

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
