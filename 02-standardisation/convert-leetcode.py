import os
import sys
import json
import random
import argparse
from subprocess import run
from datetime import datetime, UTC
from typing import Dict, Any, Optional
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets

SRC = "newfacade/LeetCodeDataset"

def lint_code(code: str) -> Optional[str]:
    result = run(
        ["ruff", "format", "-"],
        input=code.strip().encode(),
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return f"```python\n{result.stdout.decode()}```"

def convert_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single sample to the new format."""
    
    # Start with existing fields
    converted: Dict[str, Any] = {
        "conversation_id": "",
        "dataset_source": SRC,
        "original_metadata": {},
        "created_timestamp": datetime.now(UTC).isoformat(),
    }
    
    # System prompt is always the same, we don't want it
    converted["system_prompt"] = {
        "content": "",
        "metadata": {},
    }

    starter_code = sample['starter_code']

    if random.random() < 0.5:
        starter_code = starter_code.strip()
    
    # Process initial_prompt
    converted["initial_prompt"] = {
        "role": "user",
        "content": f"{sample['problem_description'].strip().replace('\n\n\xa0\n', '\n\n').replace('\n\n\n', '\n\n')}\n\nStarter code:\n\n```python\n{starter_code}\n```",
        "metadata": {}
    }
    
    # No available functions in this dataset
    converted["available_functions"] = []

    response = sample["response"]

    # Extract and lint all Python code blocks in the response
    parts = response.split("```python")
    if len(parts) > 1:
        # Process each code block
        processed_parts = [parts[0]]  # First part (before any code)
        
        for i in range(1, len(parts)):
            if "```" in parts[i]:
                code_part, after_code = parts[i].split("```", 1)
                linted_code = lint_code(code_part)
                if linted_code is None:
                    return None
                processed_parts.append(linted_code)
                processed_parts.append(after_code)
            else:
                # Malformed code block, keep as is
                processed_parts.append(parts[i])
        
        response = "".join(processed_parts)

    parts: list[Dict] = [
        {
            "type": "response",
            "content": response,
            "metadata": {}
        }
    ]

    # if answer is not None:
    #     parts.append({
    #         "type": "verifiable-responses",
    #         "answers": [],
    #     })
    
    # Process conversation branches    
    converted["conversation_branches"] = [
        {
            "messages": [
                {
                    "role": "assistant",
                    "parts": parts,
                },
            ],
        },
    ]
    
    return converted

def load_existing_metadata(output_path: Path) -> Optional[Dict[str, Any]]:
    """Load existing dataset metadata if it exists."""
    meta_file = output_path / "dataset_metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None

def save_dataset_and_metadata(dataset_dict: DatasetDict, output_path: Path, args: argparse.Namespace):
    """Save converted dataset with processing metadata."""
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    dataset_dict.save_to_disk(str(output_path))
    
    # Load existing metadata or create new
    metadata = load_existing_metadata(output_path) or {}
    
    # Create processing entry
    processing_entry = {
        "operation": f"convert_{SRC}",
        "script": f"convert_{SRC}.py",
        "timestamp": datetime.now(UTC).isoformat(),
        "input_path": args.input,
        "output_path": str(output_path),
        "num_processes": args.num_proc,
        "limit": args.limit,
        "description": f"Converted {SRC} dataset from JSON to unified chat format"
    }
    
    # Add to processing log
    if "processing_log" not in metadata:
        metadata["processing_log"] = []
    metadata["processing_log"].append(processing_entry)
    
    # Add format metadata if not already present
    if "format" not in metadata:
        metadata["format"] = "chat_format_v1"
    if "source_dataset" not in metadata:
        metadata["source_dataset"] = SRC
    if "conversion_details" not in metadata:
        metadata["conversion_details"] = {
            "conversation_type": "LEXam_open",
            "added_fields": ["system_prompt", "conversation_branches"],
            "format": "new_chat_format_with_parts"
        }
    
    # Save metadata
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset saved to {output_path}")
    print(f"Metadata saved to {metadata_file}")

def cli():
    p = argparse.ArgumentParser(description=f"Convert {SRC} dataset to unified chat format")
    p.add_argument("-i", "--input", default=None, help="Input file path. If None, will be loaded from the Hub.")
    p.add_argument("-o", "--output", required=True, help="Output directory path")
    p.add_argument("--num-proc", type=int, default=8, help="Number of processes for dataset operations")
    p.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")
    return p.parse_args()

def main():
    args = cli()
    output_path = Path(args.output)
    
    # Check if output exists
    if output_path.exists():
        response = input(f"{output_path} exists. Overwrite? [y/N]: ")
        if response.lower() != "y":
            sys.exit(0)
    
    if args.input is None:
        dataset_train = load_dataset("newfacade/LeetCodeDataset", split="train")
        dataset_test = load_dataset("newfacade/LeetCodeDataset", split="test")
        dataset = DatasetDict({"test": concatenate_datasets([dataset_test, dataset_train])})
    else:
        # Load JSON data
        print(f"Loading data from {args.input}")
        try:
            with open(args.input, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading input file: {e}")
            sys.exit(1)
        dataset = DatasetDict({'test': Dataset.from_list(data)})
    
    if not isinstance(dataset, DatasetDict):
        print(f"Unexpected dataset type.")
        sys.exit(1)
        
    data = dataset["test"]
    print(f"Loaded {len(data)} samples")

    def filter_fn(sample: Dict[str, Any]) -> bool:
        return len(sample["response"]) > 0
    
    data = data.filter(filter_fn, num_proc=args.num_proc)
    print(data)

    # def filter_fn(sample: Dict[str, Any]) -> bool:
    #     return len(sample["response"].split("```python")) != 2
    
    # data = data.filter(filter_fn, num_proc=args.num_proc)
    # print(data)
    # for x in data:
    #     with open(f"leetcode_fix/{x['question_id']}.txt", "w") as f:
    #         f.write(x["problem_description"])
    #         f.write("\n\n<SEPARATOR>\n\n")
    #         f.write(x["response"])
    # exit()

    # import re
    # PATTERN = re.compile(r"This code .* correctly")
    # def filter_fn(sample: Dict[str, Any]) -> bool:
    #     return PATTERN.search(sample["response"]) is not None
    
    # data = data.filter(filter_fn, num_proc=args.num_proc)
    # print(data)

    # os.makedirs("leetcode_fix", exist_ok=True)
    # files = os.listdir("leetcode_fix")
    # files = set([f.split(".")[0] for f in files])
    # for x in data:
    #     if x["question_id"] not in files:
    #         with open(f"leetcode_fix/{x['question_id']}.txt", "w") as f:
    #             f.write(x["problem_description"])
    #             f.write("\n\n<SEPARATOR>\n\n")
    #             f.write(x["response"])
    # exit()

    # Merge the manual fixes
    files = os.listdir("leetcode_fix")
    question_ids = set([int(f.split(".")[0]) for f in files])
    for x in data:
        if x["question_id"] in question_ids:
            with open(f"leetcode_fix/{x['question_id']}.txt", "r") as f:
                text = f.read()
                problem_description, response = text.split("\n\n<SEPARATOR>\n\n")
                x["problem_description"] = problem_description
                x["response"] = response

    # Apply limit if specified
    if args.limit and args.limit > 0:
        data = data[:args.limit]
        print(f"Limited to {len(data)} samples")
    
    # Convert samples
    print("Converting samples to new format...")
    converted_samples = []
    for i, sample in enumerate(data):
        if i % 1000 == 0:
            print(f"Processing sample {i}/{len(data)}")
        converted_samples.append(convert_sample(sample))
    
    # Create Dataset and DatasetDict
    print("Creating DatasetDict...")    
    dataset = Dataset.from_list(converted_samples)
    dataset_dict = DatasetDict({"train": dataset})
    
    print(f"Converted {len(converted_samples)} samples")
    
    # Save dataset and metadata
    save_dataset_and_metadata(dataset_dict, output_path, args)
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()