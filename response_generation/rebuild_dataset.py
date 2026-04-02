import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from response_generation.generate import load_input_dataset, reconstruct_dataset


def infer_model(output_dir, dataset):
    model_file = os.path.join(output_dir, "model_used.txt")
    if os.path.exists(model_file):
        with open(model_file, "r", encoding="utf-8") as f:
            return f.read().strip()

    if "generation_meta" in dataset.column_names and len(dataset) > 0:
        first_meta = dataset["generation_meta"][0]
        if isinstance(first_meta, dict):
            model = first_meta.get("model")
            if model:
                return model
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild a Hugging Face dataset from response_generation/responses.jsonl."
    )
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True, help="Directory containing responses.jsonl")
    parser.add_argument("--target-output-dir", type=str, default=None, help="Where to save the rebuilt dataset; defaults to --output-dir")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--model", type=str, default=None, help="Optional fallback model name/path for generation_meta")
    parser.add_argument("--temperature", type=float, default=1.0, help="Fallback temperature used when rows do not include generation_meta")
    parser.add_argument("--max-length", type=int, default=8096, help="Fallback max_length used when rows do not include generation_meta")
    args = parser.parse_args()

    output_jsonl = os.path.join(args.output_dir, "responses.jsonl")
    if not os.path.exists(output_jsonl):
        raise FileNotFoundError(f"responses.jsonl not found: {output_jsonl}")

    target_output_dir = args.target_output_dir or args.output_dir
    dataset = load_input_dataset(args.dataset_path, args.split)
    model = args.model or infer_model(args.output_dir, dataset)

    rebuilt = reconstruct_dataset(
        dataset,
        output_jsonl,
        model=model,
        temperature=args.temperature,
        max_length=args.max_length,
    )
    rebuilt.save_to_disk(target_output_dir)

    if model:
        with open(os.path.join(target_output_dir, "model_used.txt"), "w", encoding="utf-8") as f:
            f.write(model)

    print(f"Saved rebuilt dataset to {target_output_dir}")


if __name__ == "__main__":
    main()
