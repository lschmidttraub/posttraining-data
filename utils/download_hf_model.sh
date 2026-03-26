#!/bin/bash
#SBATCH --job-name=download_hf_model
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/download/hf_model_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs/download

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch $0 <model-id> [local-dir]"
  exit 1
fi

MODEL_ID="$1"
MODEL_SLUG="${MODEL_ID//\//_}"

if [ -f "${HOME}/.hf_secrets" ]; then
  source "${HOME}/.hf_secrets"
fi

SCRATCH_ROOT="${SCRATCH:-/tmp}"
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf_home}"
if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi
if [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
  export HF_TOKEN="${HUGGINGFACE_HUB_TOKEN}"
fi

LOCAL_DIR="${2:-${SCRATCH_ROOT}/models/${MODEL_SLUG}}"

mkdir -p "${HF_HOME}/hub"
mkdir -p "${LOCAL_DIR}"

srun --environment="./response_generation/env/alignment.toml" \
  --container-writable \
  --container-workdir="$PWD" \
  bash -lc "
set -euo pipefail
export HF_HOME='${HF_HOME}'
export HF_TOKEN='${HF_TOKEN:-}'
export HUGGINGFACE_HUB_TOKEN='${HUGGINGFACE_HUB_TOKEN:-}'
export HUGGINGFACE_HUB_CACHE='${HF_HOME}/hub'
export TRANSFORMERS_CACHE='${HF_HOME}/hub'
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 - <<'PY'
import importlib
import subprocess
import sys

for pkg in ('huggingface_hub', 'hf_transfer'):
    try:
        importlib.import_module(pkg)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])

from huggingface_hub import snapshot_download

local_dir = '${LOCAL_DIR}'
model_id = '${MODEL_ID}'

path = snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    max_workers=2,
    resume_download=True,
)
print(path)
PY"
