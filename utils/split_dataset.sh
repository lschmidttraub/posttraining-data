#!/bin/bash
#SBATCH --job-name=split_dataset
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/split/dataset_%j.log

# Split a local HuggingFace dataset into k chunks of at most MAX_SIZE samples.
#
# Usage:
#   sbatch utils/split_dataset.sh <input_dataset> <max_size> [output_dir] [split]
#
# Examples:
#   sbatch utils/split_dataset.sh /scratch/datasets/my-dataset 5000
#   sbatch utils/split_dataset.sh /scratch/datasets/my-dataset 5000 /scratch/datasets/chunks train

set -euo pipefail

mkdir -p logs/split

if [ "$#" -lt 2 ]; then
  echo "Usage: sbatch $0 <input_dataset> <max_size> [output_dir] [split]"
  echo ""
  echo "  input_dataset  Path to the input dataset directory"
  echo "  max_size       Maximum number of samples per chunk"
  echo "  output_dir     Directory for output chunks (default: parent of input_dataset)"
  echo "  split          Specific split to process (default: 'train' or first available)"
  exit 1
fi

INPUT_DATASET="$1"
MAX_SIZE="$2"
OUTPUT_DIR="${3:-}"
SPLIT="${4:-}"

if [ ! -d "${INPUT_DATASET}" ]; then
  echo "Error: input dataset path does not exist or is not a directory: ${INPUT_DATASET}"
  exit 1
fi

srun --environment="./response_generation/env/alignment.toml" \
  --container-writable \
  --container-workdir="$PWD" \
  bash -c "python3 -u - <<'PY'
import sys
import math
import json
from pathlib import Path
from datetime import datetime

from datasets import load_from_disk, DatasetDict

input_path = Path('${INPUT_DATASET}')
max_size = ${MAX_SIZE}
output_dir_arg = '${OUTPUT_DIR}'
split_arg = '${SPLIT}'

output_dir = Path(output_dir_arg) if output_dir_arg else input_path.parent
split_name = split_arg if split_arg else None

# Load dataset
print(f'Loading dataset from: {input_path}')
dataset = load_from_disk(str(input_path))
dataset_name = input_path.name

# Load existing metadata
metadata_file = input_path / 'dataset_metadata.json'
original_metadata = {}
if metadata_file.exists():
    try:
        with open(metadata_file, 'r') as f:
            original_metadata = json.load(f)
    except (json.JSONDecodeError, IOError):
        pass

# Resolve the target split
if hasattr(dataset, 'keys'):
    available_splits = list(dataset.keys())
    print(f'Found DatasetDict with splits: {available_splits}')
    if split_name:
        if split_name not in available_splits:
            print(f\"Error: split '{split_name}' not found. Available: {available_splits}\")
            sys.exit(1)
        target_split = split_name
    elif 'train' in available_splits:
        target_split = 'train'
    else:
        target_split = available_splits[0]
    data = dataset[target_split]
else:
    target_split = 'train'
    data = dataset

dataset_size = len(data)
k = math.ceil(dataset_size / max_size)

print(f'Split: {target_split} ({dataset_size:,} samples)')
print(f'Max chunk size: {max_size:,}')
print(f'Number of chunks: {k}')

if k <= 1:
    print('Dataset already fits in a single chunk - nothing to do.')
    sys.exit(0)

output_dir.mkdir(parents=True, exist_ok=True)
start_idx = 0

for i in range(k):
    end_idx = min(start_idx + max_size, dataset_size)
    chunk_data = data.select(range(start_idx, end_idx))
    chunk_dataset = DatasetDict({'train': chunk_data})

    suffix = str(i + 1)
    output_name = f'{dataset_name}-chunk{suffix}'
    output_path = output_dir / output_name

    output_path.mkdir(parents=True, exist_ok=True)
    chunk_dataset.save_to_disk(str(output_path))

    processing_entry = {
        'operation': 'dataset_max_size_split',
        'script': 'utils/split_dataset.sh',
        'timestamp': datetime.now().isoformat(),
        'input_path': str(input_path),
        'output_path': str(output_path),
        'source_split': target_split,
        'chunk_index': i + 1,
        'total_chunks': k,
        'max_size': max_size,
        'samples': len(chunk_data),
        'start_index': start_idx,
        'end_index': end_idx - 1,
    }

    metadata = {
        **original_metadata,
        'processing_log': original_metadata.get('processing_log', []) + [processing_entry],
    }
    with open(output_path / 'dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'  [{i+1}/{k}] {output_path} ({len(chunk_data):,} samples)')
    start_idx = end_idx

print(f'Done. Created {k} chunks in {output_dir}')
PY"
