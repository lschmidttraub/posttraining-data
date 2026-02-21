#!/bin/bash

# 1. Define the list of jobs as formatted strings: "ModelName nnodes dp_size tp_size"
JOBS=(
    "Qwen/Qwen2.5-0.5B-Instruct 1 4 1"
    "Qwen/Qwen2.5-1.5B-Instruct 1 4 1"
    "Qwen/Qwen3-0.6B 1 4 1"
    "Qwen/Qwen3-1.7B 1 4 1"
    "Qwen/Qwen3-4B-Instruct-2507 1 1 1"
    "Qwen/Qwen3-8B 1 4 1"
)

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
#SBATCH --time=09:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1

cd $SCRATCH/posttraining-data/response_generation

# Execute the generation script with the dynamically assigned variables
srun --environment=activeuf --container-workdir="$SCRATCH/posttraining-data/response_generation" \\
    bash -c "unset SSL_CERT_FILE && python -u run_generation.py --model '${MODEL}' --slurm-nodes ${NNODES} --dp-size ${DP} --tp-size ${TP}"
EOF

done

echo "✅ All jobs have been successfully queued!"