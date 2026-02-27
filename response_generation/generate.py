import os
import json
import asyncio
import argparse
from openai import AsyncOpenAI
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm_asyncio

async def writer_task(queue, filepath):
    """Listens to the queue and writes outputs to the JSONL file immediately."""
    with open(filepath, "a", encoding="utf-8") as f:
        while True:
            item = await queue.get()
            if item is None:  # Poison pill to stop the writer
                break
            f.write(json.dumps(item) + "\n")
            f.flush() # Ensure it's immediately written to disk
            queue.task_done()

async def get_response(idx, prompt, client, model, max_length, temperature, semaphore, queue):
    """Fetches the response and immediately puts it in the write queue."""
    async with semaphore:
        try:
            res = await client.chat.completions.create(
                model=model,
                messages=prompt,
                max_tokens=max_length,
                temperature=temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = res.choices[0].message.content
        except Exception as e:
            print(f"Error for index {idx}: {e}")
            content = ""

        # Push directly to the writer queue, dropping logprobs entirely
        await queue.put({
            "index": idx,
            "response": content
        })

async def main(args):
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")

    # 1. Load Dataset
    if os.path.exists(args.dataset_path):
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset(args.dataset_path, split="train")

    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, "responses.jsonl")

    # 2. Fast Resume logic (Read existing JSONL)
    processed_indices = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    processed_indices.add(data["index"])
        print(f"✅ Resuming from checkpoint: Found {len(processed_indices)} already processed items.")

    if len(processed_indices) >= len(dataset):
        print("Already finished processing the entire dataset.")
        return

    # 3. Extract and Filter Prompts
    print(f"Extracting prompts from {len(dataset)} samples...")
    all_prompts = [sample["chosen"][:-1] for sample in dataset]
    
    valid_indices = []
    if args.max_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        too_long_count = 0
        for i, prompt in enumerate(all_prompts):
            if i in processed_indices: 
                continue # Skip already processed
            text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            if len(tokenizer.encode(text, add_special_tokens=False)) < args.max_tokens:
                valid_indices.append(i)
            else:
                too_long_count += 1
        print(f"Filtered out {too_long_count} prompts exceeding {args.max_tokens} tokens.")
    else:
        valid_indices = [i for i in range(len(all_prompts)) if i not in processed_indices]

    if not valid_indices:
        print("No valid prompts left to process.")
        return

    # 4. Setup Async Queue and Concurrency
    queue = asyncio.Queue()
    semaphore = asyncio.Semaphore(args.concurrent)
    
    # Start the background writer task
    writer = asyncio.create_task(writer_task(queue, output_jsonl))

    # 5. Create and Run Tasks
    print(f"🚀 Processing {len(valid_indices)} prompts...")
    tasks = [
        get_response(idx, all_prompts[idx], client, args.model, args.max_length, args.temperature, semaphore, queue) 
        for idx in valid_indices
    ]
    
    # Use as_completed to avoid gathering huge arrays of objects into memory
    for f in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        await f

    # 6. Shutdown Writer gracefully
    await queue.put(None)
    await writer

    print("✅ All requests completed and written to JSONL.")

    # 7. (Optional) Final Reconstruction into Hugging Face Dataset
    print(f"💾 Reconstructing and saving final dataset to {args.output_dir}")
    
    # Load everything back into ordered memory just once at the end
    final_responses = [""] * len(dataset)
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                final_responses[data["index"]] = data["response"]

    dataset = dataset.add_column("response", final_responses)
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
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    asyncio.run(main(args))