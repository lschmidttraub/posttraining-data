#!/bin/bash

# 1. Define the list of jobs as formatted strings: "ModelName nnodes dp_size tp_size"
JOBS=(
    "Qwen/Qwen2.5-0.5B-Instruct 1 4 1"
    "Qwen/Qwen2.5-1.5B-Instruct 1 4 1"
    "Qwen/Qwen3-0.6B 1 4 1"
    "Qwen/Qwen3-1.7B 1 4 1"
    "Qwen/Qwen3-4B-Instruct-2507 1 1 1" #3 gpus unutilized. 
    "Qwen/Qwen3-8B 1 4 1"
    "Qwen/Qwen3-32B 2 4 1"
    "Qwen/Qwen3-30B-A3B-Instruct-2507 2 1 4"
    "Qwen/Qwen3-Omni-30B-A3B-Instruct 2 1 4"
    "Qwen/Qwen3-Omni-30B-A3B-Instruct 2 1 4"
    "Qwen/Qwen3-Next-80B-A3B-Instruct 2 1 4"
    "Qwen/Qwen3-235B-A22B-Instruct-2507 4 1 8"
)
BASE_OUTPUT_DIR=""
JOB_TIME="09:00:00"

# 2. Ensure the log directory exists before submitting
mkdir -p ./logs/generation

# TODO: adapt for multinode setup. 
# 3. Loop through each defined job
for ENTRY in "${JOBS[@]}"; do
    
    # Extract the tuple values into separate variables
    read -r MODEL NNODES DP TP <<< "$ENTRY"
    
    # Create a safe filename by replacing slashes with underscores
    SAFE_MODEL_NAME=$(echo "$MODEL" | tr '/' '_')
    
    echo "Submitting generation job for: $MODEL (Nodes: $NNODES, DP: $DP, TP: $TP)"
    
    # 4. Pass the SLURM directives and execution command directly to sbatch
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=gen_${SAFE_MODEL_NAME}
#SBATCH -A a-infra01-1
#SBATCH --output=./logs/generation/${SAFE_MODEL_NAME}_%j.log
#SBATCH --time=${JOB_TIME}
#SBATCH --partition=normal
#SBATCH --nodes=1

cd $SCRATCH/posttraining-data/response_generation

# Execute the generation script with the dynamically assigned variables
srun --environment=activeuf --container-workdir="$SCRATCH/posttraining-data/response_generation" \\
    bash -c "unset SSL_CERT_FILE && python -u run_generation.py --base-output-dir '${BASE_OUTPUT_DIR}' --job-time ${JOB_TIME} --model '${MODEL}' --slurm-nodes ${NNODES} --dp-size ${DP} --tp-size ${TP}"
EOF

done

echo "✅ All jobs have been successfully queued!"