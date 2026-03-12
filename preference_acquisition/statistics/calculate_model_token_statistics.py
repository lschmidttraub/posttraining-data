import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from collections import defaultdict
from datasets import load_from_disk
from transformers import AutoTokenizer

# Paths
DATASET_PATH = "/iopsstor/scratch/cscs/dmelikidze/posttraining-data/response_annotation/datasets/all_completions_annotated"
OUTPUT_DIR = "/iopsstor/scratch/cscs/dmelikidze/posttraining-data/preference_acquisition/"
TOKENIZER_MODEL_PATH = "/iopsstor/scratch/cscs/dmelikidze/sft-checkpoints/apertus1-base-sft-stage1/global_step_2406/huggingface"

def generate_75_percent_plot(global_stats, metric_name, output_filename, xlabel):
    model_avgs = {m: np.mean(scores) for m, scores in global_stats.items()}
    sorted_models = sorted(model_avgs.keys(), key=lambda k: model_avgs[k], reverse=False)
    
    custom_plot_stats = []
    for model in sorted_models:
        scores = np.array(global_stats[model])
        p_low, med, p_high = np.percentile(scores, [12.5, 50, 87.5]) # Combined percentile call
        
        custom_plot_stats.append({
            'label': model, 'mean': np.mean(scores), 'med': med,
            'q1': p_low, 'q3': p_high, 'whislo': p_low, 'whishi': p_high, 'fliers': []
        })
    
    fig_height = max(6, len(sorted_models) * 0.35)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    bp = ax.bxp(custom_plot_stats, vert=False, patch_artist=True, showmeans=True, showfliers=False,
                whiskerprops=dict(linewidth=0), capprops=dict(linewidth=0),
                boxprops=dict(color='black'), medianprops=dict(color='black', linewidth=1.5),
                meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='red', markersize=5))
    
    for box in bp['boxes']:
        box.set_facecolor('mediumseagreen')
        box.set_alpha(0.6)
    
    fmt = "{:.0f}" if "Log" not in metric_name else "{:.2f}"
    for i, model in enumerate(sorted_models):
        ax.text(model_avgs[model], i + 1.25, f"Avg: {fmt.format(model_avgs[model])}", 
                va='center', ha='center', fontsize=8, color='black')

    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_title(f"Core {metric_name} Distribution", fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.legend(handles=[
        Patch(facecolor='mediumseagreen', edgecolor='black', alpha=0.6, label='Middle 75%'),
        Line2D([0], [0], color='black', lw=1.5, label='Median'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='red', markeredgecolor='black', markersize=7, label='Average')
    ], loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    num_proc = os.cpu_count() or 1
    
    dataset = load_from_disk(DATASET_PATH)["train"]
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_PATH)

    # Optimization 1: Batched mapping is significantly faster
    def add_token_info_batch(examples):
        # Batch tokenization is faster than per-example
        tokenized = tokenizer(examples["response"], add_special_tokens=False, padding=False, truncation=False)
        lengths = [max(1, len(t)) for t in tokenized["input_ids"]]
        return {
            "token_length": lengths,
            "log_token_length": np.log10(lengths).tolist()
        }

    print(f"Calculating token lengths using {num_proc} CPUs (Batched)...")
    dataset_with_lengths = dataset.map(
        add_token_info_batch,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        keep_in_memory=False, 
        desc="Tokenizing responses"
    )

    print("Aggregating statistics per model...")
    # Optimization 3: Convert only needed columns to pandas/numpy for vectorized grouping
    df = dataset_with_lengths.to_pandas()
    
    raw_token_stats = {}
    log_token_stats = {}
    
    # Using Pandas groupby is significantly faster than a Python for-loop
    grouped = df.groupby("model")
    for model_name, group in grouped:
        raw_token_stats[model_name] = group["token_length"].values
        log_token_stats[model_name] = group["log_token_length"].values

    # --- SAVE STATISTICS ---
    stats_output_path = os.path.join(OUTPUT_DIR, "model_token_statistics.txt")
    sorted_models = sorted(raw_token_stats.keys(), key=lambda m: np.mean(raw_token_stats[m]))

    with open(stats_output_path, "w") as f:
        header = f"{'Model':<35} | {'Raw Avg':<8} | {'Raw Std':<8} | {'Log Avg':<8} | {'Log Std':<8}"
        f.write(header + "\n" + "-" * 75 + "\n")
        for model in sorted_models:
            r, l = raw_token_stats[model], log_token_stats[model]
            line = f"{model:<35} | {np.mean(r):<8.0f} | {np.std(r):<8.0f} | {np.mean(l):<8.2f} | {np.std(l):<8.2f}"
            f.write(line + "\n")
            
    # --- GENERATE PLOTS ---
    generate_75_percent_plot(raw_token_stats, "Token Length", "all_models_75_percent_tokens.png", "Number of Tokens")
    generate_75_percent_plot(log_token_stats, "Log10 Token Length", "all_models_75_percent_log_tokens.png", "Log10(Number of Tokens)")

if __name__ == "__main__":
    main()