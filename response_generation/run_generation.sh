#!/bin/bash

# Format: "ModelName TotalNodes Workers NodesPerWorker DP TP DisableOCF(true/false)"
JOBS=(
    "Qwen/Qwen2.5-0.5B-Instruct 1 1 1 4 1 false sglang"
    "Qwen/Qwen2.5-1.5B-Instruct 1 1 1 4 1 false sglang"
    "Qwen/Qwen3-0.6B 1 1 1 4 1 false sglang"
    "Qwen/Qwen3-1.7B 1 1 1 4 1 false sglang"
    "Qwen/Qwen3-4B-Instruct-2507 1 1 1 4 1 false sglang"
    "Qwen/Qwen3-8B 1 1 1 4 1 false sglang"
    "Qwen/Qwen3-32B 4 4 1 4 1 false sglang"
    "Qwen/Qwen3-30B-A3B-Instruct-2507 4 4 1 1 4 false sglang"
    "Qwen/Qwen3-Omni-30B-A3B-Instruct 4 4 1 1 4 false sglang"
    "Qwen/Qwen3-Next-80B-A3B-Instruct 4 4 1 1 4 false sglang"
    "Qwen/Qwen3-235B-A22B-Instruct-2507 32 16 2 1 8 true sglang"
    "microsoft/Phi-4-mini-instruct 1 1 1 4 1 false sglang"
    "mistralai/Mistral-Small-24B-Instruct-2501 4 4 1 1 4 false sglang"
    "mistralai/Mixtral-8x22B-Instruct-v0.1 16 8 2 1 8 true sglang"
    "mistralai/Ministral-3-3B-Instruct-2512 1 1 1 4 1 false sglang"
    "mistralai/Ministral-3-8B-Instruct-2512 1 1 1 4 1 false sglang"
    "mistralai/Ministral-3-14B-Instruct-2512 1 1 1 4 1 false sglang"
    "arcee-ai/Trinity-Mini 1 1 1 4 1 false vllm"
    "arcee-ai/Trinity-Nano-Preview 1 1 1 4 1 false vllm"
    "HuggingFaceTB/SmolLM3-3B 1 1 1 4 1 false sglang"
    "utter-project/EuroLLM-1.7B-Instruct 1 1 1 4 1 false sglang"
    "utter-project/EuroLLM-9B-Instruct-2512 1 1 1 4 1 false sglang"
    "utter-project/EuroLLM-22B-Instruct-2512 1 1 1 4 1 false sglang"
)

BASE_OUTPUT_DIR="./datasets/inference_results_final5"
JOB_TIME="12:00:00"

ACCOUNT="infra01"
RESERVATION="PA-2338-RL"
WORKING_DIR="$SCRATCH/posttraining-data/response_generation"

mkdir -p ./logs/generation

for ENTRY in "${JOBS[@]}"; do
    read -r MODEL NNODES WORKERS NPW DP TP DOCF FRAMEWORK <<< "$ENTRY"
    SAFE_MODEL_NAME=$(echo "$MODEL" | tr '/' '_')

    OCF_FLAG=""
    if [ "$DOCF" = "true" ]; then OCF_FLAG="--disable-ocf"; fi

    env -i PATH=$PATH HOME=$HOME TERM=$TERM USER=$USER LOGNAME=$USER SCRATCH=$SCRATCH \
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=gen_${SAFE_MODEL_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --output=./logs/generation/${SAFE_MODEL_NAME}_%j.log
#SBATCH --time=${JOB_TIME}
#SBATCH --reservation=${RESERVATION}                 # Uncomment if you have a reservation to use
#SBATCH --partition=normal
#SBATCH --nodes=1

# Direct directory change
cd ${WORKING_DIR}

# Using --container-workdir to chdir inside the container as well
srun --environment=activeuf --container-writable --container-workdir="${WORKING_DIR}" \\
    bash -c "unset SSL_CERT_FILE && python -u run_generation.py \\
    --base-output-dir '${BASE_OUTPUT_DIR}' \\
    --model '${MODEL}' \\
    --slurm-nodes ${NNODES} \\
    --workers ${WORKERS} \\
    --nodes-per-worker ${NPW} \\
    --dp-size ${DP} \\
    --tp-size ${TP} \\
    --framework '${FRAMEWORK}' \\
    ${OCF_FLAG}"
EOF
done

echo "✅ All jobs submitted."