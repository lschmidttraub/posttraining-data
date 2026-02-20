#!/bin/bash

# 1. Define the list of models to process
MODELS=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B-Instruct-2507"
    "Qwen/Qwen3-8B"

    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
)

# 2. Ensure the log directory exists before submitting
mkdir -p ./logs/generation

# 3. Loop through each model and submit an independent job
for MODEL in "${MODELS[@]}"; do
    
    # Create a safe filename by replacing slashes with underscores (e.g., Qwen_Qwen3-8B)
    SAFE_MODEL_NAME=$(echo "$MODEL" | tr '/' '_')
    
    echo "Submitting generation job for: $MODEL"
    
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
# Execute the generation script (assuming run_generation.py is the orchestrator)
srun --environment=activeuf --container-workdir="$SCRATCH/posttraining-data/response_generation" \
    bash -c "unset SSL_CERT_FILE && python -u run_generation.py --model '${MODEL}'"
EOF

done

echo "✅ All jobs have been successfully queued!"