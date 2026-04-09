#!/bin/bash

# Format: "ModelName TotalNodes Workers NodesPerWorker DP TP DisableOCF(true/false) Framework NoReasoningKwargs(true/false)"
JOBS=(
  # "Qwen/Qwen3-8B 1 1 1 4 1 false sglang false"
  # "Qwen/Qwen3-32B 1 1 1 4 1 false sglang false"
  # "${SCRATCH}/models/Qwen_Qwen3.5-397B-A17B 32 8 4 1 16 true vllm false"
  # "${SCRATCH}/models/zai-org_GLM-5 32 4 8 1 32 true sglang true"
  "${SCRATCH}/models/zai-org_GLM-5-FP8 4 1 4 1 16 true sglang true"
  # "$SCRATCH/models/nvidia_GLM-5-NVFP4 8 4 2 1 8 true sglang true"
)

CATEGORY="${CATEGORY:-}"
INPUT_DATASET="${INPUT_DATASET:-$SCRATCH/datasets/chunked/science/science-chunk1}"

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
JOB_TIME="${JOB_TIME:-6:00:00}"
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
#SBATCH --job-name=$(basename $INPUT_DATASET)
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

EOF
done

echo "✅ All jobs submitted."
