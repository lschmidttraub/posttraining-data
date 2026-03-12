#!/bin/bash

# Fixed model and cluster configuration for Qwen3-235B
MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507"
NNODES=4
WORKERS=2
NPW=2
DP=1
TP=8
FRAMEWORK="sglang"
# FRAMEWORK="vllm"
OCF_FLAG="--disable-ocf"

# Array of absolute paths to your datasets
DATASETS=(
    # $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/Mistral-Small-24B-Instruct-2501
    # $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/Qwen2.5-0.5B-Instruct
    # $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/Qwen3-1.7B          
    # $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/Qwen3-32B          
    # $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/Qwen3-Next-80B-A3B-Instruct
    # $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/Mixtral-8x22B-Instruct-v0.1   
    # $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/Qwen2.5-1.5B-Instruct 
    # $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/Qwen3-235B-A22B-Instruct-2507 
    # $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/Qwen3-4B-Instruct-2507 
    # $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/Qwen3-Omni-30B-A3B-Instruct
    # $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/Phi-4-mini-instruct           
    # $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/Qwen3-0.6B           
    # $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/Qwen3-30B-A3B-Instruct-2507  
    # $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/Qwen3-8B
    # $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/EuroLLM-1.7B-Instruct 
    # $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/Ministral-3-8B-Instruct-2512 
    # $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/SmolLM3-3B 
    # $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/EuroLLM-22B-Instruct-2512     
    # $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/Trinity-Mini
    # $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/EuroLLM-9B-Instruct-2512             
    # $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/Trinity-Nano-Preview
    # $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/Ministral-3-14B-Instruct-2512          
    # $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/Ministral-3-3B-Instruct-2512  
    # $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/Mistral-Large-3-675B-Instruct-2512
    $SCRATCH/posttraining-data/response_generation/datasets/inference_results_final/Qwen3.5-397B-A17B
)

BASE_OUTPUT_DIR="$SCRATCH/posttraining-data/response_annotation/datasets/inference_results_final-test"
JOB_TIME="12:00:00"

ACCOUNT="infra01"
RESERVATION="PA-2338-RL"
WORKING_DIR="$SCRATCH/posttraining-data/response_annotation"

mkdir -p ./logs/annotation

for DATASET_PATH in "${DATASETS[@]}"; do
    # Extract just the folder name of the dataset for cleaner job names and logs
    SAFE_DATASET_NAME=$(basename "$DATASET_PATH")

    env -i PATH=$PATH HOME=$HOME TERM=$TERM USER=$USER LOGNAME=$USER SCRATCH=$SCRATCH \
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ann_${SAFE_DATASET_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --output=./logs/annotation/${SAFE_DATASET_NAME}_%j.log
#SBATCH --time=${JOB_TIME}
#SBATCH --reservation=${RESERVATION}                 # Uncomment if you have a reservation to use
#SBATCH --partition=normal
#SBATCH --nodes=1

# Direct directory change
cd ${WORKING_DIR}

# Using --container-workdir to chdir inside the container as well
srun --environment=activeuf --container-writable --container-workdir="${WORKING_DIR}" \\
    bash -c "unset SSL_CERT_FILE && python -u run_annotation.py \\
    --base-output-dir '${BASE_OUTPUT_DIR}' \\
    --dataset '${DATASET_PATH}' \\
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