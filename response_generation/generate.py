import os
import json
import random
import asyncio
import argparse
import httpx
import uvloop
from openai import AsyncOpenAI, APIStatusError, APIConnectionError, APITimeoutError
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm_asyncio

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

def has_saved_response(response):
    """Return True when a saved response contains non-whitespace text."""
    return isinstance(response, str) and bool(response.strip())

def sanitize_messages(messages):
    """Strip all non-standard fields from messages to avoid vLLM/Mistral validation errors."""
    # Standard OpenAI chat message fields per role
    ALLOWED_KEYS = {
        "system": {"role", "content", "name"},
        "user": {"role", "content", "name"},
        "assistant": {"role", "content", "name", "tool_calls", "refusal"},
        "tool": {"role", "content", "tool_call_id"},
    }
    DEFAULT_ALLOWED = {"role", "content", "name"}
    
    sanitized = []
    for msg in messages:
        role = msg.get("role", "user")
        allowed = ALLOWED_KEYS.get(role, DEFAULT_ALLOWED)
        clean_msg = {k: v for k, v in msg.items() if k in allowed and v is not None}
        # Ensure role and content are always present
        clean_msg["role"] = role
        if "content" not in clean_msg:
            clean_msg["content"] = msg.get("content", "")
        sanitized.append(clean_msg)
    return sanitized

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

def _is_retryable(exc):
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code in RETRYABLE_STATUS_CODES:
        return True
    return False

async def get_response(idx, prompt, client, model, max_length, temperature, semaphore, queue, use_reasoning=True, max_retries=7):
    """Fetches the response with exponential backoff on transient errors."""
    async with semaphore:
        prompt = sanitize_messages(prompt)
        kwargs = dict(
            model=model,
            messages=prompt,
            max_tokens=max_length,
            temperature=temperature,
        )
        if use_reasoning:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        content = ""
        for attempt in range(max_retries + 1):
            try:
                res = await client.chat.completions.create(**kwargs)
                content = res.choices[0].message.content
                break
            except Exception as e:
                if _is_retryable(e) and attempt < max_retries:
                    delay = min(2 ** attempt, 120) + random.uniform(0, 1)
                    print(f"[retry {attempt+1}/{max_retries}] index {idx}: {e} — retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                else:
                    print(f"Error for index {idx} (after {attempt+1} attempts): {e}")
                    break

        await queue.put({
            "index": idx,
            "response": content
        })

async def main(args):
    # 1. UNLOCK HTTPX LIMITS
    # Ensure the HTTP connection pool matches the exact size of your semaphore
    custom_limits = httpx.Limits(
        max_connections=args.concurrent, 
        max_keepalive_connections=args.concurrent
    )
    
    # 10 minutes timeout for handling massive batch processing delays
    custom_timeout = httpx.Timeout(7200.0) 
    
    http_client = httpx.AsyncClient(limits=custom_limits, timeout=custom_timeout)

    client = AsyncOpenAI(
        base_url=args.base_url, 
        api_key="EMPTY",
        http_client=http_client
    )

    # 2. Load Dataset
    if os.path.exists(args.dataset_path):
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset(args.dataset_path, split=args.split)
        
    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, "responses.jsonl")

    # 3. Fast Resume logic (Read existing outputs)
    existing_responses = {}
    if os.path.exists(output_jsonl):
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    existing_responses[data["index"]] = data.get("response", "")
        print(f"✅ Resuming from checkpoint: Found {len(existing_responses)} saved items in responses.jsonl.")
    elif args.retry_existing and "response" in dataset.column_names:
        existing_responses = {
            idx: response
            for idx, response in enumerate(dataset["response"])
        }
        print(f"✅ Found {len(existing_responses)} saved items in the dataset response column.")

    processed_indices = set(existing_responses)
    empty_response_indices = {
        idx for idx, response in existing_responses.items()
        if not has_saved_response(response)
    }

    if args.retry_existing and empty_response_indices:
        processed_indices -= empty_response_indices
        print(f"🔁 Retrying {len(empty_response_indices)} items with empty saved responses.")

    if len(processed_indices) >= len(dataset):
        print("Already finished processing the entire dataset.")
        # Make sure to close the client before returning early
        await http_client.aclose()
        return

    # 4. Extract and Filter Prompts
    all_prompts = dataset[args.prompt_column_name]
    if isinstance(all_prompts[0], str):
        all_prompts = [[{"role": "user", "content": p}] for p in all_prompts]
    if args.remove_last_message:
        print("⚠️ Removing last message from each prompt as per --remove-last-message flag.")
        all_prompts = [p[:-1] if len(p) > 1 else p for p in all_prompts]
    
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
        await http_client.aclose()
        return

    # 5. Setup Async Queue and Concurrency
    queue = asyncio.Queue()
    semaphore = asyncio.Semaphore(args.concurrent)
    
    # Start the background writer task
    writer = asyncio.create_task(writer_task(queue, output_jsonl))

    # 6. Create and Run Tasks
    print(f"🚀 Processing {len(valid_indices)} prompts...")
    use_reasoning = not args.no_reasoning_kwargs
    tasks = [
        get_response(idx, all_prompts[idx], client, args.model, args.max_length, args.temperature, semaphore, queue, use_reasoning, args.max_retries) 
        for idx in valid_indices
    ]
    
    # Use as_completed to avoid gathering huge arrays of objects into memory
    for f in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        await f

    # 7. Shutdown Writer gracefully
    await queue.put(None)
    await writer

    # Close the custom http client properly
    await http_client.aclose()

    print("✅ All requests completed and written to JSONL.")

    # 8. (Optional) Final Reconstruction into Hugging Face Dataset
    print(f"💾 Reconstructing and saving final dataset to {args.output_dir}")
    
    # Load everything back into ordered memory just once at the end
    if "response" in dataset.column_names:
        final_responses = list(dataset["response"])
    else:
        final_responses = [""] * len(dataset)
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                final_responses[data["index"]] = data["response"]

    if "response" in dataset.column_names:
        dataset = dataset.remove_columns("response")
    dataset = dataset.add_column("response", final_responses)
    dataset.save_to_disk(args.output_dir)

    with open(os.path.join(args.output_dir, "model_used.txt"), "w") as f:
        f.write(args.model)

if __name__ == "__main__":
    # Install uvloop for maximum performance before anything else
    # python generate.py --dataset-path=allenai/Dolci-Instruct-DPO --output-dir=./datasets/tmpbax3 --model=moonshotai/Kimi-K2.5-dmelikidze --concurrent=250 --base-url=http://172.28.33.32:8080/v1
    uvloop.install()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--prompt-column-name", type=str, default="prompt", help="Name of the column containing the prompt/messages")
    parser.add_argument("--remove-last-message", action="store_true", help="Whether to remove the last message from the conversation history, e.g. if you take it from a 'chosen' column")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--split", type=str, default="train")
    
    # Increased default concurrency to better saturate the 16 nodes
    parser.add_argument("--concurrent", type=int, default=200)
    parser.add_argument("--base-url", type=str, default="https://serving.swissai.cscs.ch/")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--no-reasoning-kwargs", action="store_true", help="Disable passing chat_template_kwargs (needed for Mistral tokenizers)")
    parser.add_argument("--retry-existing", action="store_true", help="Retry samples whose saved response is empty in responses.jsonl or an existing response column")
    parser.add_argument("--max-retries", type=int, default=7, help="Max retries per request on transient errors (503, 429, timeouts, etc.)")
    args = parser.parse_args()

    asyncio.run(main(args))