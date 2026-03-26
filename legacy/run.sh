INPUT_DATASET_DIR="<input_dataset_directory>"
# Will save multiple intermediate datasets and the final dataset in this directory
OUTPUT_DIR="<output_directory>"
LOG_DIR=$OUTPUT_DIR/logs

SLURM_PARTITION=normal
SLURM_RESERVATION=PA-2338-RL

DATASET_NAME=debug

REFERENCE_MODEL=apertus-8b-sft
REFERENCE_MODEL_NAME_OR_PATH="\${artifacts_dir}/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-7-ln-v2-ademamix/checkpoints/7fea1f8c44336360/checkpoint-8925"
MAX_SEQUENCE_LENGTH=4096

# ===== DATASET PREPROCESSING =====
cd $POSTTRAINING_DATA_DIR

# Activate 
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ ! -d "${SCRIPT_DIR}/venv" ]; then
    echo "Environment not set up"
    echo "Please create a virtual environment based on the requirements.txt file in ${SCRIPT_DIR} at ${SCRIPT_DIR}/venv/bin/activate"
    exit 1
fi
source venv/bin/activate

# 1. Standardise the dataset
python 02-standardisation/convert-standard-format.py $INPUT_DATASET_DIR $OUTPUT_DIR/01-standard-format

# 2. Decontaminate the dataset
./04-decontamination/submit-decontamination.sh $OUTPUT_DIR/01-standard-format $OUTPUT_DIR/02-decontaminated $SLURM_RESERVATION


# ===== PREPARE FOR TRAINING
# 3. Filter the dataset for the model
# * Can be replaced with submit-parallel-decontamination to make faster
python 09-filter-for-ref-model/filter_dataset_for_ref_model_swissaiformat.py --dataset $OUTPUT_DIR/02-decontaminated --model_name_or_path $REFERENCE_MODEL_NAME_OR_PATH --max-seq-len $MAX_SEQUENCE_LENGTH --output_path $OUTPUT_DIR/03-filtered-for-ref-model