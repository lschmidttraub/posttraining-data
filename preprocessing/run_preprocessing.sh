#!/bin/bash

DATASET="${DATASET:-salesforce/xlam-function-calling-60k}"
DATASETS="${DATASETS:-salesforce/xlam-function-calling-60k,salesforce/xlam-function-calling-60k}"
MAPPER="${MAPPER:-Salesforce/xlam-function-calling-60k,salesforce/xlam-function-calling-60k}"
MAPPERS="${MAPPERS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/datasets/preprocessed/$(basename "${DATASET}")}"
SPLIT="${SPLIT:-train}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
NUM_PROC="${NUM_PROC:-}"
UPLOAD_TO_HUB="${UPLOAD_TO_HUB:-0}"
HUB_DATASET_ID="${HUB_DATASET_ID:-}"
HUB_PRIVATE="${HUB_PRIVATE:-0}"
JOB_TIME="${JOB_TIME:-02:00:00}"
ACCOUNT="${ACCOUNT:-infra01}"
LOGS_DIR="${LOGS_DIR:-./logs/preprocessing}"
JOB_NAME="${JOB_NAME:-prep_$(echo "${MAPPER}" | tr '/-' '__')}"
SAFE_NAME="$(echo "${DATASET}" | tr '/-' '__')"

mkdir -p "${LOGS_DIR}"

DATASET_FLAGS=()
if [ -n "${DATASETS}" ]; then
  IFS=',' read -r -a DATASET_LIST <<<"${DATASETS}"
  for ITEM in "${DATASET_LIST[@]}"; do
    DATASET_FLAGS+=("--dataset" "${ITEM}")
  done
else
  DATASET_FLAGS+=("--dataset" "${DATASET}")
fi

MAPPER_FLAGS=()
if [ -n "${MAPPERS}" ]; then
  IFS=',' read -r -a MAPPER_LIST <<<"${MAPPERS}"
  for ITEM in "${MAPPER_LIST[@]}"; do
    MAPPER_FLAGS+=("--mapper" "${ITEM}")
  done
else
  MAPPER_FLAGS+=("--mapper" "${MAPPER}")
fi

CLI_ARGS=()
CLI_ARGS+=("${DATASET_FLAGS[@]}")
CLI_ARGS+=("${MAPPER_FLAGS[@]}")
CLI_ARGS+=("--output-dir" "${OUTPUT_DIR}")
if [ -n "${SPLIT}" ]; then
  CLI_ARGS+=("--split" "${SPLIT}")
fi
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
