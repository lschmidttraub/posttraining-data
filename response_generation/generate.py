import os
import json
import asyncio
import argparse
import multiprocessing
from openai import AsyncOpenAI
from datasets import load_from_disk, load_dataset, DatasetDict
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from transformers import AutoTokenizer

async def main(args):
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")

    if os.path.exists(args.dataset_path):
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset(args.dataset_path, split="train")

    print(f"Extracting prompts from {len(dataset)} samples...")
    all_prompts = []
    for sample in tqdm(dataset):
        messages = [{"role": m["role"], "content": m["content"]} for m in sample["chosen"][:-1]]
        all_prompts.append(messages)
    
    
    valid_indices = []
    if args.max_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        too_long_count = 0
        for i, prompt in enumerate(all_prompts):
            text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            if len(tokenizer.encode(text, add_special_tokens=False)) < args.max_tokens:
                valid_indices.append(i)
            else:
                too_long_count += 1
        print(f"Filtered out {too_long_count} prompts exceeding {args.max_tokens} tokens.")
    else:
        valid_indices = list(range(len(all_prompts)))

    semaphore = asyncio.Semaphore(args.concurrent)
    
    responses = [""] * len(dataset)
    logprobs = [[]] * len(dataset)

    async def get_response(idx):
        async with semaphore:
            try:
                kwargs = {}
                if not "mistral" in args.model.lower():
                    kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

                res = await client.chat.completions.create(
                    model=args.model,
                    messages=all_prompts[idx],
                    logprobs=True,
                    top_logprobs=1,
                    max_tokens=model_to_max_tokens.get(args.model, args.max_length),
                    **kwargs
                )
                content = res.choices[0].message.content
                lp = [{"token": l.token, "logprob": l.logprob} for l in res.choices[0].logprobs.content]
                return idx, content, lp
            except Exception as e:
                # Print error but keep going
                print(f"Error for index {idx}: {e}")
                return idx, "", []

    batch_size = 25000
    for i in range(0, len(valid_indices), batch_size):
        chunk_indices = valid_indices[i : i + batch_size]
        tasks = [get_response(idx) for idx in chunk_indices]
        
        print(f"🚀 Processing Batch {i//batch_size + 1} (indices {i} to {i+len(chunk_indices)})")
        results = await tqdm_asyncio.gather(*tasks)
        
        for idx, resp, lp in results:
            responses[idx] = resp
            logprobs[idx] = lp

        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_{i + len(chunk_indices)}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        num_covered = i + len(chunk_indices)
        checkpoint_ds = dataset.select(range(num_covered)) # Keep all rows if you want, or slice
        checkpoint_ds = checkpoint_ds.add_column("response", responses[:num_covered])
        checkpoint_ds = checkpoint_ds.add_column("logprobs", logprobs[:num_covered])
        checkpoint_ds.save_to_disk(checkpoint_dir)
        
    dataset = dataset.add_column("response", responses)
    dataset = dataset.add_column("logprobs", logprobs)

    print(f"💾 Saving final dataset to {args.output_dir}")
    dataset.save_to_disk(args.output_dir)

    with open(os.path.join(args.output_dir, "model_used.txt"), "w") as f:
        f.write(args.model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--concurrent", type=int, default=1000)
    parser.add_argument("--base-url", type=str, default="https://serving.swissai.cscs.ch/")
    args = parser.parse_args()

    asyncio.run(main(args))