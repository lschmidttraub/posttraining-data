import os
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

DATASET_PATH = "/iopsstor/scratch/cscs/dmelikidze/posttraining-data/response_annotation/datasets/all_completions_annotated"
OUTPUT_DIR = "/iopsstor/scratch/cscs/dmelikidze/posttraining-data/preference_acquisition/"
OUTPUT_PLOT_PATH = os.path.join(OUTPUT_DIR, "all_models_wide_histograms.png")

def process_shard(shard_index, num_shards):
    """
    Worker function: Loads a shard of the dataset and aggregates scores.
    Replaces any score less than 1 with exactly 1.
    """
    dataset = load_from_disk(DATASET_PATH)["train"]
    shard = dataset.shard(num_shards=num_shards, index=shard_index)

    local_stats = defaultdict(list)
    for example in shard:
        # Enforce the minimum score of 1
        score = max(1.0, float(example["score"]))
        local_stats[example["model"]].append(score)
        
    return dict(local_stats)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    num_proc = os.cpu_count() or 1
    print(f"Using {num_proc} CPUs for parallel data aggregation...")

    global_stats = defaultdict(list)

    # 1. MULTIPROCESSING
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        futures = [executor.submit(process_shard, i, num_proc) for i in range(num_proc)]
        for future in as_completed(futures):
            local_stats = future.result()
            for model, scores in local_stats.items():
                global_stats[model].extend(scores)

    # Sort models by average score for a logical visual progression from left to right
    model_avgs = {m: np.mean(scores) for m, scores in global_stats.items()}
    sorted_models = sorted(model_avgs.keys(), key=lambda k: model_avgs[k], reverse=False)
    num_models = len(sorted_models)
    
    print(f"Data aggregated. Generating wide histogram plot for {num_models} models...")
    
    # 2. PLOTTING: 1 Row, N Columns. 
    fig, axes = plt.subplots(nrows=1, ncols=num_models, figsize=(3 * num_models, 3.5), sharey=True, squeeze=False)
    
    for i, model in enumerate(sorted_models):
        ax = axes[0, i]
        scores = np.array(global_stats[model])
        
        # Plot the individual histogram
        ax.hist(scores, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        
        # Format the model title to handle long names
        formatted_title = model.replace("-", "-\n") if len(model) > 15 else model
        ax.set_title(formatted_title, fontsize=10, fontweight='bold')
        
        # Coordinate axis setup
        ax.set_xlabel("r", fontsize=10, fontweight='bold')
        ax.set_xlim(0.8, 5.2)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        if i == 0:
            ax.set_ylabel("Frequency", fontsize=10)
        
        # 3. TEXTUAL DATA: N is removed, only Avg and Std remain
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        stats_text = f"Avg: {avg_score:.2f}\nStd: {std_score:.2f}"
        
        # Place box in the top left corner of each subplot
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=8, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Adjust layout
    plt.subplots_adjust(wspace=0.1)
    
    print(f"Saving wide plot to {OUTPUT_PLOT_PATH}...")
    plt.savefig(OUTPUT_PLOT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print("Done!")

if __name__ == "__main__":
    main()