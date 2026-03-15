"""
Stage 3: Compute reference log-probabilities for chosen and rejected.

Computes per-sequence log-probabilities for chosen and rejected completions
using the reference model. These precomputed logprobs avoid needing two
model copies during DPO/QRPO training.

Input:  HuggingFace dataset with 'chosen', 'rejected', and 'prompt_messages'
        columns (output of stage 2).
Output: Dataset with added columns:
        - ref_chosen_logprob, chosen_length
        - ref_rejected_logprob, rejected_length

Usage:
    python compute_logprobs.py \
        --dataset-path /path/to/dataset_with_completions \
        --output-dir /path/to/output \
        --model-name-or-path /path/to/model
"""

import argparse
import json
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def compute_logprob(model, tokenizer, prompt_messages, completion_text, device, max_seq_len):
    """
    Compute the log-probability of a completion given a prompt.

    Args:
        prompt_messages: list of message dicts (the conversation up to the assistant turn)
        completion_text: the assistant response string

    Returns:
        (logprob: float, completion_length: int)
    """
    # Build full conversation: prompt + assistant response
    full_messages = prompt_messages + [{"role": "assistant", "content": completion_text}]

    # Tokenize full conversation
    full_ids = tokenizer.apply_chat_template(full_messages, tokenize=True, return_tensors="pt").to(device)

    # Tokenize prompt only (with generation prompt to get the assistant header tokens)
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    seq_len = full_ids.shape[1]
    if seq_len > max_seq_len:
        return None, None

    prompt_len = prompt_ids.shape[1]
    completion_length = seq_len - prompt_len

    if completion_length <= 0:
        return 0.0, 0

    with torch.no_grad():
        outputs = model(input_ids=full_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Shift: predict token t from logits at position t-1
    shift_logits = logits[:, :-1, :]  # (1, seq_len-1, vocab)
    shift_labels = full_ids[:, 1:]     # (1, seq_len-1)

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    # Only sum over completion tokens (positions prompt_len-1 to seq_len-2 in shifted space)
    completion_log_probs = token_log_probs[:, prompt_len - 1:]
    total_logprob = completion_log_probs.sum().item()

    return total_logprob, completion_length


def process_batch(model, tokenizer, batch_prompts, batch_completions, device, max_seq_len):
    """Process a batch of (prompt, completion) pairs. Returns list of (logprob, length)."""
    results = []
    for prompt_msgs, completion in zip(batch_prompts, batch_completions):
        lp, cl = compute_logprob(model, tokenizer, prompt_msgs, completion, device, max_seq_len)
        results.append((lp, cl))
    return results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

    # Handle partition slicing
    if args.partition_start is not None and args.partition_end is not None:
        total = len(dataset)
        start = args.partition_start
        end = min(args.partition_end, total)
        print(f"Processing partition [{start}, {end}) out of {total}")
        dataset = dataset.select(range(start, end))

    print(f"Dataset size: {len(dataset)}")

    print(f"Loading model from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    chosen_logprobs = []
    chosen_lengths = []
    rejected_logprobs = []
    rejected_lengths = []

    for i in tqdm(range(len(dataset)), desc="Computing logprobs"):
        row = dataset[i]

        # Extract prompt messages
        prompt_msgs = json.loads(row["prompt_messages"])

        # Chosen: extract final assistant message
        chosen_msgs = row["chosen"]
        chosen_text = chosen_msgs[-1]["content"]
        c_lp, c_len = compute_logprob(model, tokenizer, prompt_msgs, chosen_text, device, args.max_seq_len)
        chosen_logprobs.append(c_lp)
        chosen_lengths.append(c_len)

        # Rejected: extract final assistant message
        rejected_msgs = row["rejected"]
        rejected_text = rejected_msgs[-1]["content"]
        r_lp, r_len = compute_logprob(model, tokenizer, prompt_msgs, rejected_text, device, args.max_seq_len)
        rejected_logprobs.append(r_lp)
        rejected_lengths.append(r_len)

    # Add columns
    dataset = dataset.add_column("ref_chosen_logprob", chosen_logprobs)
    dataset = dataset.add_column("chosen_length", chosen_lengths)
    dataset = dataset.add_column("ref_rejected_logprob", rejected_logprobs)
    dataset = dataset.add_column("rejected_length", rejected_lengths)

    print(f"Saving to {args.output_dir}")
    dataset.save_to_disk(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute reference logprobs for preference dataset")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to input HF dataset (output of stage 2)")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save output dataset")
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Reference model path")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Max sequence length (skip longer sequences)")
    parser.add_argument("--partition-start", type=int, default=None, help="Start index for dataset partition")
    parser.add_argument("--partition-end", type=int, default=None, help="End index for dataset partition")
    args = parser.parse_args()
    main(args)
