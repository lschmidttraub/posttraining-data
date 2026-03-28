#!/bin/bash

# has to be one of the keys in MAPPER_REGISTRY
CATEGORY="${CATEGORY:-tool_calling}"
# has to be a comma-separated list of dataset names in the category. If empty, use all datasets in the category.
DATASETS="${DATASETS:-}"
NAME="${NAME:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/datasets/preprocessed/${CATEGORY}${NAME:+/${NAME}}}"
# mapping hyperparameters
BATCH_SIZE="${BATCH_SIZE:-1000}"
NUM_PROC="${NUM_PROC:-}"

# HF upload 
UPLOAD_TO_HUB="${UPLOAD_TO_HUB:-0}"
HUB_DATASET_ID="${HUB_DATASET_ID:-}"
HUB_PRIVATE="${HUB_PRIVATE:-0}"

# SLURM job parameters
JOB_TIME="${JOB_TIME:-12:00:00}"
ACCOUNT="${ACCOUNT:-infra01}"
LOGS_DIR="${LOGS_DIR:-./logs/preprocessing}"
JOB_NAME="${JOB_NAME:-prep_${CATEGORY}}"
SAFE_NAME="${CATEGORY}"

mkdir -p "${LOGS_DIR}"

CLI_ARGS=("--category" "${CATEGORY}")
if [ -n "${DATASETS}" ]; then
  IFS=',' read -r -a DATASET_LIST <<<"${DATASETS}"
  for ITEM in "${DATASET_LIST[@]}"; do
    CLI_ARGS+=("--dataset" "${ITEM}")
  done
fi
CLI_ARGS+=("--output-dir" "${OUTPUT_DIR}")
CLI_ARGS+=("--batch-size" "${BATCH_SIZE}")

NUM_PROC_FLAG=""
if [ -n "${NUM_PROC}" ]; then
  NUM_PROC_FLAG="--num-proc ${NUM_PROC}"
fi

UPLOAD_FLAGS=""
if [ "${UPLOAD_TO_HUB}" = "1" ]; then
  UPLOAD_FLAGS="--upload-to-hub --hub-dataset-id '${HUB_DATASET_ID}'"
  if [ "${HUB_PRIVATE}" = "1" ]; then
    UPLOAD_FLAGS="${UPLOAD_FLAGS} --hub-private"
  fi
fi

CLI_ARGS_STRING=""
printf -v CLI_ARGS_STRING "%q " "${CLI_ARGS[@]}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --output=${LOGS_DIR}/${SAFE_NAME}_%j.log
#SBATCH --time=${JOB_TIME}
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --mail-type=END,FAIL

srun --environment="./response_generation/env/alignment.toml" --container-writable --container-workdir="$PWD" \\
    bash -c "unset SSL_CERT_FILE && python -u -m preprocessing.run \\
    ${CLI_ARGS_STRING} \\
    ${NUM_PROC_FLAG} \\
    ${UPLOAD_FLAGS}"
EOF

echo "Submitted preprocessing job."
