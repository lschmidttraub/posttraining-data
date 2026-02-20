from datasets import load_from_disk

dataset = load_from_disk("/iopsstor/scratch/cscs/dmelikidze/posttraining-data/response_generation/output/Qwen3-14B")

print(dataset)
print(dataset["response"][53])
print(dataset["prompt"][53])