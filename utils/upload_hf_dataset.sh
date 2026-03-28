#!/bin/bash
#SBATCH --job-name=upload_hf_dataset
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/upload/hf_dataset_%j.log

set -euo pipefail

mkdir -p logs/upload


LOCAL_DATASET_PATH="$1"
REPO_ID="$2"
HUB_PRIVATE="${3:-True}"

if [ ! -d "${LOCAL_DATASET_PATH}" ]; then
  echo "Local dataset path does not exist or is not a directory: ${LOCAL_DATASET_PATH}"
  exit 1
fi

mkdir -p "${HF_HOME:-${HOME}/.cache/huggingface}/hub"

srun --environment="./response_generation/env/alignment.toml" \
  --container-writable \
  --container-workdir="$PWD" \
  bash -lc "
set -euo pipefail
export HF_HOME='${HF_HOME:-${HOME}/.cache/huggingface}'
export HF_TOKEN='${HF_TOKEN:-}'
export HUGGINGFACE_HUB_CACHE=\"\${HF_HOME}/hub\"
python3 - <<'PY'
import importlib
import subprocess
import sys

from datasets import load_from_disk

local_dataset_path = '${LOCAL_DATASET_PATH}'
repo_id = '${REPO_ID}'
hub_private = ${HUB_PRIVATE}

dataset = load_from_disk(local_dataset_path)
dataset.push_to_hub(repo_id, private=hub_private)
print(f'Uploaded {local_dataset_path} to {repo_id} (private={hub_private})')
PY"
