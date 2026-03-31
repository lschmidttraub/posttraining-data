#!/bin/bash

# Format: "ModelName TotalNodes Workers NodesPerWorker DP TP DisableOCF(true/false) Framework NoReasoningKwargs(true/false)"
JOBS=(
  # "Qwen/Qwen3-8B 1 1 1 4 1 false sglang false"
  # "Qwen/Qwen3-32B 1 1 1 4 1 false sglang false"
  # "${SCRATCH}/models/Qwen_Qwen3.5-397B-A17B 32 8 4 1 16 true vllm false"
  "${SCRATCH}/models/zai-org_GLM-5 32 4 8 1 32 true sglang true"
)

CATEGORY="${CATEGORY:-}"
INPUT_DATASET="${INPUT_DATASET:-$SCRATCH/datasets/chunked/science/science-chunk100}"

# If category is set, we preprocess the category and use the result for generation
if [ -n "${CATEGORY}" ]; then
  DATASET_NAME="${CATEGORY}"
  PREPROCESSED_DATASET_DIR="${PREPROCESSED_DATASET_DIR:-/tmp/datasets/preprocessed/${CATEGORY}}"
  mkdir -p "${PREPROCESSED_DATASET_DIR}"
  DATASET_FLAGS=(
    "--preprocess"
    "--preprocess-category" "${CATEGORY}"
    "--preprocessed-dataset-dir" "${PREPROCESSED_DATASET_DIR}"
  )
else
  DATASET_NAME="$(basename "${INPUT_DATASET}")"
  DATASET_FLAGS=("--dataset" "${INPUT_DATASET}")
fi

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$SCRATCH/datasets/completions/$DATASET_NAME}"
JOB_TIME="12:00:00"
SPLIT="train"

ACCOUNT="infra01"
LOGS_DIR="./logs/generation"

mkdir -p $LOGS_DIR/client $LOGS_DIR/server

printf -v DATASET_FLAGS_STRING "%q " "${DATASET_FLAGS[@]}"


for ENTRY in "${JOBS[@]}"; do
  read -r MODEL NNODES WORKERS NPW DP TP DOCF FRAMEWORK GLM <<<"$ENTRY"
  SAFE_MODEL_NAME=$(basename $MODEL)

  OCF_FLAG=""
  if [ "$DOCF" = "true" ]; then OCF_FLAG="--disable-ocf"; fi

  GLM_FLAG=""
  if [ "$GLM" = "true" ]; then GLM_FLAG="--glm --pre-launch-cmds 'PIP_CONSTRAINT= pip install blobfile'"; fi

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=gen_${SAFE_MODEL_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --output=${LOGS_DIR}/client/${SAFE_MODEL_NAME}_%j.log
#SBATCH --time=${JOB_TIME}
#SBATCH --partition=normal
#SBATCH --nodes=1

#SBATCH hetjob
#SBATCH --nodes=${NNODES}


uv run python -u response_generation/run_generation.py \\
    ${DATASET_FLAGS_STRING} \\
    --base-output-dir '${BASE_OUTPUT_DIR}' \\
    --logs-dir '${LOGS_DIR}/server' \\
    --model '${MODEL}' \\
    --slurm-nodes ${NNODES} \\
    --workers ${WORKERS} \\
    --nodes-per-worker ${NPW} \\
    --dp-size ${DP} \\
    --tp-size ${TP} \\
    --framework '${FRAMEWORK}' \\
    --job-time '${JOB_TIME}' \\
    --account ${ACCOUNT} \\
    --split '${SPLIT}' \\
    --client-hetgroup 0 \\
    --server-hetgroup 1 \\
    ${OCF_FLAG} --enforce-eager ${GLM_FLAG}

# srun --overlap --het-group=0 --environment="./response_generation/env/alignment.toml" --container-writable --container-workdir="$PWD" \\
#     bash -c "unset SSL_CERT_FILE && python -u response_generation/run_generation.py \\
#     ${DATASET_FLAGS_STRING} \\
#     --base-output-dir '${BASE_OUTPUT_DIR}' \\
#     --logs-dir '${LOGS_DIR}/server' \\
#     --model '${MODEL}' \\
#     --slurm-nodes ${NNODES} \\
#     --hetgroup 1 \\
#     --workers ${WORKERS} \\
#     --nodes-per-worker ${NPW} \\
#     --dp-size ${DP} \\
#     --tp-size ${TP} \\
#     --framework '${FRAMEWORK}' \\
#     --job-time '${JOB_TIME}' \\
#     --account ${ACCOUNT} \\
#     --split '${SPLIT}' \\
#     ${OCF_FLAG} --enforce-eager ${GLM_FLAG}"

EOF
done

echo "✅ All jobs submitted."
