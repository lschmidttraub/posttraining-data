import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
from datasets import load_from_disk
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

DATASET_PATH = "/iopsstor/scratch/cscs/dmelikidze/posttraining-data/response_annotation/datasets/all_completions_annotated"
OUTPUT_DIR = "/iopsstor/scratch/cscs/dmelikidze/posttraining-data/preference_acquisition/"
OUTPUT_PLOT_PATH = os.path.join(OUTPUT_DIR, "all_models_75_percent_core.png")

def process_shard(shard_index, num_shards):
    """
    Worker function: Loads a shard of the dataset and aggregates scores.
    Replaces any score less than 1 with exactly 1.
    """
    dataset = load_from_disk(DATASET_PATH)["train"]
    shard = dataset.shard(num_shards=num_shards, index=shard_index)

    local_stats = defaultdict(list)
    for example in shard:
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

    # Calculate averages and sort models
    model_avgs = {m: np.mean(scores) for m, scores in global_stats.items()}
    sorted_models = sorted(model_avgs.keys(), key=lambda k: model_avgs[k], reverse=False)
    
    # 2. CALCULATE CUSTOM PERCENTILES FOR 75% BOX
    custom_plot_stats = []
    for model in sorted_models:
        scores = np.array(global_stats[model])
        avg = np.mean(scores)
        med = np.median(scores)
        
        # Middle 75% boundaries
        p_low = np.percentile(scores, 12.5)
        p_high = np.percentile(scores, 87.5)
        
        custom_plot_stats.append({
            'label': model,
            'mean': avg,
            'med': med,
            'q1': p_low,       # Left edge of the box
            'q3': p_high,      # Right edge of the box
            'whislo': p_low,   # Cap whiskers to the box edges
            'whishi': p_high,
            'fliers': []       # Empty list to ensure no outliers are drawn
        })
    
    print(f"Generating 75% core distribution plot for {len(sorted_models)} models...")
    
    # 3. PLOTTING
    fig_height = max(6, len(sorted_models) * 0.35)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    
    # Use ax.bxp (Boxplot with custom stats) to draw the custom 75% boxes
    bp = ax.bxp(custom_plot_stats, vert=False, patch_artist=True, showmeans=True, showfliers=False,
                whiskerprops=dict(linewidth=0),   
                capprops=dict(linewidth=0),       
                boxprops=dict(facecolor='mediumseagreen', color='black', alpha=0.6),
                medianprops=dict(color='black', linewidth=1.5),
                meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='red', markersize=5))
    
    for box in bp['boxes']:
        box.set_facecolor('mediumseagreen')
        box.set_alpha(0.6)
    
    # Add textual annotations for the average score
    for i, model in enumerate(sorted_models):
        avg = model_avgs[model]
        ax.text(avg, i + 1.25, f"Avg: {avg:.2f}", va='center', ha='center', fontsize=8, color='black')

    # Axes setup
    ax.set_xlabel("r", fontsize=12, fontweight='bold')
    ax.set_title("Core Score Distribution", fontsize=14, fontweight='bold')
    ax.set_xlim(0.8, 5.2)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    # 4. LEGEND CREATION
    legend_elements = [
        Patch(facecolor='mediumseagreen', edgecolor='black', alpha=0.6, label='Middle 75% of Scores'),
        Line2D([0], [0], color='black', lw=1.5, label='Median Score'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='red', markeredgecolor='black', markersize=7, label='Average Score')
    ]
    
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.08), 
              ncol=3, fontsize=9, framealpha=0.9, edgecolor='gray')

    plt.tight_layout()
    
    print(f"Saving plot to {OUTPUT_PLOT_PATH}...")
    plt.savefig(OUTPUT_PLOT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print("Done!")

if __name__ == "__main__":
    main()