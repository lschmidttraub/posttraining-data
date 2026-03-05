import os
import statistics
from datasets import load_from_disk
from transformers import AutoTokenizer

# 1. Setup paths
dataset_path = "/iopsstor/scratch/cscs/dmelikidze/posttraining-data/response_generation/datasets/preferences/Qwen3-32B_vs_0.6B"
tokenizer_path = "/users/dmelikidze/projects/posttraining/run/artifacts/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-7-ln-v2-ademamix/checkpoints/7fea1f8c44336360/checkpoint-8925"

# 2. Load Tokenizer & Dataset
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
dataset = load_from_disk(dataset_path)

def get_completion_len(messages):
    """
    Calculates the token length of only the last assistant message 
    formatted within the chat template.
    """
    if not messages:
        return 0
    
    # Full conversation length
    full_text_tokens = tokenizer.apply_chat_template(messages, tokenize=True)
    
    # Length of everything EXCEPT the last assistant message
    prompt_tokens = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    
    return len(full_text_tokens) - len(prompt_tokens)

def process_batch(examples):
    chosen_lens = [get_completion_len(c) for c in examples["chosen"]]
    rejected_lens = [get_completion_len(r) for r in examples["rejected"]]
    return {"chosen_len": chosen_lens, "rejected_len": rejected_lens}

# 3. Apply Mapping (Multi-core)
processed_ds = dataset.map(
    process_batch,
    batched=True,
    num_proc=os.cpu_count(),
    desc="Calculating completion lengths"
)

# 4. Filter out any samples where completion is empty/error
filtered_ds = processed_ds.filter(
    lambda x: x["chosen_len"] > 0 and x["rejected_len"] > 0,
    num_proc=os.cpu_count(),
    desc="Filtering valid lengths"
)

# 5. Calculate and Print Advanced Stats
def print_stats(ds, name="Dataset"):
    c_lens = ds["chosen_len"]
    r_lens = ds["rejected_len"]
    n_samples = len(ds)
    
    # Base Aggregates
    mean_yc = sum(c_lens) / n_samples
    mean_yr = sum(r_lens) / n_samples
    med_yc = statistics.median(c_lens)
    med_yr = statistics.median(r_lens)
    
    # Option 1: 1 / HM(mean_yc, mean_yr)
    hm_of_means = (2 * mean_yc * mean_yr) / (mean_yc + mean_yr)
    inv_hm_means = 1 / hm_of_means
    
    # Option 2: 1 / HM(med_yc, med_yr)
    hm_of_medians = (2 * med_yc * med_yr) / (med_yc + med_yr)
    inv_hm_medians = 1 / hm_of_medians
    
    print(f"\n=== Statistics: {name} ===")
    print(f"Total Samples:         {n_samples}")
    print(f"Avg (Mean) Chosen:     {mean_yc:.2f} tokens")
    print(f"Avg (Mean) Rejected:   {mean_yr:.2f} tokens")
    print(f"Median Chosen:         {med_yc:.2f} tokens")
    print(f"Median Rejected:       {med_yr:.2f} tokens")
    
    print("\n--- Option 1: Using Arithmetic Means ---")
    print(f"Harmonic Mean of (Mean Y_c, Mean Y_r): {hm_of_means:.2f}")
    
    print("\n--- Option 2: Using Medians ---")
    print(f"Harmonic Mean of (Med Y_c, Med Y_r):   {hm_of_medians:.2f}")
    print("=" * 35)

if isinstance(filtered_ds, dict):
    for split in filtered_ds.keys():
        print_stats(filtered_ds[split], split)
else:
    print_stats(filtered_ds)