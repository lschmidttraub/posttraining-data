#!/bin/bash

# Format: "ModelName TotalNodes Workers NodesPerWorker DP TP DisableOCF(true/false) Framework NoReasoningKwargs(true/false)"
JOBS=(
  # "Qwen/Qwen3-8B 1 1 1 4 1 false sglang false"
  # "Qwen/Qwen3-32B 1 1 1 4 1 false sglang false"
  # "${SCRATCH}/models/Qwen_Qwen3.5-397B-A17B 32 8 4 1 16 true vllm false"
  "${SCRATCH}/models/zai-org_GLM-5 32 4 8 1 32 true sglang true"
)

INPUT_DATASET="${INPUT_DATASET:-Salesforce/xlam-function-calling-60k}"
INPUT_DATASETS="${INPUT_DATASETS:-}"
PREPROCESS="${PREPROCESS:-1}"
PREPROCESS_MAPPER="${PREPROCESS_MAPPER:-Salesforce/xlam-function-calling-60k}"
PREPROCESS_MAPPERS="${PREPROCESS_MAPPERS:-}"
PREPROCESSED_DATASET_DIR="${PREPROCESSED_DATASET_DIR:-$SCRATCH/datasets/preprocessed/$(basename "$INPUT_DATASET")}"
PREPROCESS_BATCH_SIZE="${PREPROCESS_BATCH_SIZE:-1000}"
PREPROCESS_NUM_PROC="${PREPROCESS_NUM_PROC:-}"
if [ "$PREPROCESS" -eq 1 ]; then
  DATASET_NAME=$(basename "$PREPROCESSED_DATASET_DIR")
else
  DATASET_NAME=$(basename "$INPUT_DATASET")
fi
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$SCRATCH/datasets/completions/$DATASET_NAME}"
REMOVE_LAST_MESSAGE=0 # Set to 1 if you want to remove the last message from the conversation history, e.g. if you take it from a "chosen" column
JOB_TIME="12:00:00"
SPLIT="train"

ACCOUNT="infra01"
LOGS_DIR="./logs/generation"

mkdir -p $LOGS_DIR

DATASET_FLAGS=()
if [ -n "${INPUT_DATASETS}" ]; then
  IFS=',' read -r -a INPUT_DATASET_LIST <<<"${INPUT_DATASETS}"
  for ITEM in "${INPUT_DATASET_LIST[@]}"; do
    DATASET_FLAGS+=("--dataset" "${ITEM}")
  done
else
  DATASET_FLAGS+=("--dataset" "${INPUT_DATASET}")
fi
printf -v DATASET_FLAGS_STRING "%q " "${DATASET_FLAGS[@]}"

PREPROCESS_MAPPER_FLAGS=()
if [ -n "${PREPROCESS_MAPPERS}" ]; then
  IFS=',' read -r -a PREPROCESS_MAPPER_LIST <<<"${PREPROCESS_MAPPERS}"
  for ITEM in "${PREPROCESS_MAPPER_LIST[@]}"; do
    PREPROCESS_MAPPER_FLAGS+=("--preprocess-mapper" "${ITEM}")
  done
else
  PREPROCESS_MAPPER_FLAGS+=("--preprocess-mapper" "${PREPROCESS_MAPPER}")
fi
printf -v PREPROCESS_MAPPER_FLAGS_STRING "%q " "${PREPROCESS_MAPPER_FLAGS[@]}"

for ENTRY in "${JOBS[@]}"; do
  read -r MODEL NNODES WORKERS NPW DP TP DOCF FRAMEWORK GLM <<<"$ENTRY"
  SAFE_MODEL_NAME=$(basename $MODEL)

  OCF_FLAG=""
  if [ "$DOCF" = "true" ]; then OCF_FLAG="--disable-ocf"; fi

  GLM_FLAG=""
  if [ "$GLM" = "true" ]; then GLM_FLAG="--glm --pre-launch-cmds 'PIP_CONSTRAINT= pip install blobfile'"; fi

  REMOVE_LAST_MESSAGE_FLAG=""
  if [ "$REMOVE_LAST_MESSAGE" -eq 1 ]; then REMOVE_LAST_MESSAGE_FLAG="--remove-last-message"; fi

  PREPROCESS_FLAG=""
  if [ "$PREPROCESS" -eq 1 ]; then
    PREPROCESS_FLAG="--preprocess ${PREPROCESS_MAPPER_FLAGS_STRING} --preprocessed-dataset-dir '${PREPROCESSED_DATASET_DIR}' --preprocess-batch-size ${PREPROCESS_BATCH_SIZE}"
    if [ -n "${PREPROCESS_NUM_PROC}" ]; then
      PREPROCESS_FLAG="${PREPROCESS_FLAG} --preprocess-num-proc ${PREPROCESS_NUM_PROC}"
    fi
  fi

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=gen_${SAFE_MODEL_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --output=${LOGS_DIR}/client/${SAFE_MODEL_NAME}_%j.log
#SBATCH --time=${JOB_TIME}
#SBATCH --partition=normal
#SBATCH --nodes=1

srun --environment="./response_generation/env/alignment.toml" --container-writable --container-workdir="$PWD" \\
    bash -c "unset SSL_CERT_FILE && python -u response_generation/run_generation.py \\
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
    ${OCF_FLAG} ${REASONING_FLAG} ${REMOVE_LAST_MESSAGE_FLAG} ${PREPROCESS_FLAG} --enforce-eager ${GLM_FLAG}
EOF
done

echo "✅ All jobs submitted."
