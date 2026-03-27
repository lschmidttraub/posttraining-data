import argparse
import os

from datasets import DatasetDict, load_dataset, load_from_disk

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map a dataset into the standardized preprocessing schema.")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name or local dataset path")
    parser.add_argument("--mapper", required=True, choices=sorted(MAPPERS), help="Mapper function to apply")
    parser.add_argument("--output-dir", required=True, help="Directory where the processed dataset will be saved")
    parser.add_argument("--split", default=None, help="Optional split name when loading from the Hub")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for dataset.map")
    parser.add_argument("--num-proc", type=int, default=None, help="Optional number of worker processes for dataset.map")
    parser.add_argument("--upload-to-hub", action="store_true", help="Upload the processed dataset to the Hugging Face Hub")
    parser.add_argument("--hub-dataset-id", default=None, help="Target Hugging Face dataset repo, e.g. org/name")
    parser.add_argument("--hub-private", action="store_true", help="Create or update the Hub dataset as private")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapper = MAPPERS[args.mapper]
    dataset = load_input_dataset(args.dataset, args.split)

    processed = DatasetDict()
    for split_name, split_dataset in dataset.items():
        processed[split_name] = split_dataset.map(
            mapper,
            batched=True,
            with_indices=True,
            batch_size=args.batch_size,
            num_proc=args.num_proc,
            remove_columns=split_dataset.column_names,
            desc=f"Preprocessing {split_name}",
        ).select_columns(STANDARD_COLUMNS)

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
