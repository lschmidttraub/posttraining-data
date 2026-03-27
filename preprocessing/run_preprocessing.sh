#!/bin/bash

DATASET="${DATASET:-Salesforce/xlam-function-calling-60k}"
MAPPER="${MAPPER:-Salesforce/xlam-function-calling-60k}"
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
    --dataset '${DATASET}' \\
    --mapper '${MAPPER}' \\
    --output-dir '${OUTPUT_DIR}' \\
    --split '${SPLIT}' \\
    --batch-size ${BATCH_SIZE} \\
    ${NUM_PROC_FLAG} \\
    ${UPLOAD_FLAGS}"
EOF

echo "Submitted preprocessing job for ${DATASET}."
