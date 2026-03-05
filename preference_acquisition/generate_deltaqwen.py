import os
from datasets import load_from_disk, DatasetDict

# 1. Setup
NUM_CPUS = os.cpu_count()
print(f"Using {NUM_CPUS} CPUs for batched processing.")

dataset = load_from_disk("/iopsstor/scratch/cscs/dmelikidze/posttraining-data/response_annotation/datasets/all_completions_annotated")

# 2. Batched Extraction Function
def extract_preferences_from_window(batch):
    """
    Takes exactly 24 rows at a time. Finds the indices for the chosen/rejected 
    models, scrubs the keys, and outputs a single aggregated row.
    """
    out = {"prompt_id": [], "chosen": [], "rejected": []}
    models = batch["model"]
    
    # check that all prompt_ids in this batch are the same (sanity check)
    if len(set(batch["prompt_id"])) != 1:
        print(f"Error: Found a batch with multiple prompt_ids: {set(batch['prompt_id'])}")
        exit(-1)
    
    # Ensure both models exist in this specific 24-row window
    if "Qwen3-32B" in models and "Qwen3-0.6B" in models:
        chosen_idx = models.index("Qwen3-32B")
        rejected_idx = models.index("Qwen3-0.6B")

        # Populate the output arrays
        out["prompt_id"].append(batch["prompt_id"][chosen_idx])
        
        out["chosen"].append(
            batch["prompt"][chosen_idx] + [{"role": "assistant", "content": batch["response"][chosen_idx]}]
        )
        out["rejected"].append(
            batch["prompt"][rejected_idx] + [{"role": "assistant", "content": batch["response"][rejected_idx]}]
        )
        
    return out

# 3. Execute Parallel Batching
print("Processing dataset in 24-row windows...")
processed_dataset = dataset["train"].map(
    extract_preferences_from_window,
    batched=True,
    batch_size=24,        # Force the map function to look at exactly 24 rows at once
    num_proc=1,    # important!
    remove_columns=dataset["train"].column_names, # Remove the old 24-row columns
    desc="Extracting pairs"
)

# 4. Wrap in a DatasetDict to create the "train_split"
final_dataset_dict = DatasetDict({
    "train_split": processed_dataset
})

print(f"Created preference dataset with {len(processed_dataset)} examples in 'train_split'.")
print(final_dataset_dict["train_split"][0])

# 5. Save
output_path = "/iopsstor/scratch/cscs/dmelikidze/posttraining-data/preference_acquisition/datasets/deltaqwen_preferences"
final_dataset_dict.save_to_disk(output_path)
print("Dataset saved successfully.")