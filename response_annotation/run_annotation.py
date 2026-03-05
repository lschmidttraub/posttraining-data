import os
from datasets import load_from_disk, DatasetDict

# 1. Setup
NUM_CPUS = os.cpu_count()
print(f"Using {NUM_CPUS} CPUs for batched processing.")

dataset = load_from_disk("/iopsstor/scratch/cscs/dmelikidze/posttraining-data/response_annotation/datasets/all_completions_annotated")

# 2. Batched Extraction Function (Fixed)
def extract_preferences_from_window(batch):
    out = {"prompt_id": [], "chosen": [], "rejected": []}
    models = batch["model"]
    
    # Ensure both models exist in this specific 24-row window
    if "Qwen3-32B" in models and "Qwen3-0.6B" in models:
        chosen_idx = models.index("Qwen3-32B")
        rejected_idx = models.index("Qwen3-0.6B")
        
        prompt_data = batch["prompt"][chosen_idx]
        cleaned_prompt = []
        
        # FIX: Handle Hugging Face's weird "dict of lists" format
        if isinstance(prompt_data, dict):
            num_messages = len(prompt_data.get("role", []))
            for i in range(num_messages):
                cleaned_prompt.append({
                    "role": prompt_data["role"][i],
                    "content": prompt_data["content"][i]
                })
        # Fallback: Handle standard "list of dicts" just in case
        elif isinstance(prompt_data, list):
            for msg in prompt_data:
                if "role" in msg and "content" in msg:
                    cleaned_prompt.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        
        # Populate the output arrays
        out["prompt_id"].append(batch["prompt_id"][chosen_idx])
        
        out["chosen"].append(
            cleaned_prompt + [{"role": "assistant", "content": batch["response"][chosen_idx]}]
        )
        out["rejected"].append(
            cleaned_prompt + [{"role": "assistant", "content": batch["response"][rejected_idx]}]
        )
        
    return out

# 3. Execute Parallel Batching
print("Processing dataset in 24-row windows...")
processed_dataset = dataset["train"].map(
    extract_preferences_from_window,
    batched=True,
    batch_size=24,        
    num_proc=NUM_CPUS,    
    remove_columns=dataset["train"].column_names, 
    desc="Extracting pairs"
)

# 4. Wrap in a DatasetDict for "train_split"
final_dataset_dict = DatasetDict({
    "train_split": processed_dataset
})

print(f"Created preference dataset with {len(processed_dataset)} examples in 'train_split'.")
# Print the first row's chosen messages to PROVE the prompt is actually there this time
print("\nVerifying first row 'chosen' messages:")
print(final_dataset_dict["train_split"][0]["chosen"])

# 5. Save
output_path = "/iopsstor/scratch/cscs/dmelikidze/posttraining-data/preference_acquisition/datasets/deltaqwen_preferences"
final_dataset_dict.save_to_disk(output_path)
print("\nDataset saved successfully.")