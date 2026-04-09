import os
import re
import json
import asyncio
import argparse
import time
import httpx
import uvloop
from openai import AsyncOpenAI
from datasets import load_from_disk, load_dataset
from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm_asyncio


def has_saved_response(response):
    """Return True when a saved response contains non-whitespace text."""
    return isinstance(response, str) and bool(response.strip())

def parse_thinking(content):
    """Split a <think>...</think> block from the final answer.

    Returns (thinking, answer). If no thinking tags are present, thinking is
    an empty string and the full content is returned as the answer.

    Handles three cases:
    - Standard: <think>...</think>answer
    - Missing opening tag (GLM): thinking...</think>answer
    - No tags at all: entire content is the answer
    """
    if not content:
        return "", ""
    # Standard case: <think>...</think>
    match = re.search(r"<think>(.*?)</think>(.*)", content, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    # Missing opening <think> tag (e.g. GLM models)
    match = re.search(r"^(.*?)</think>(.*)", content, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", content.strip()

def sanitize_messages(messages):
    """Strip all non-standard fields from messages to avoid vLLM validation errors."""
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


async def get_response(idx, prompt, client, model, max_length, temperature, semaphore, queue, max_retries=3):
    """Fetches the response and immediately puts it in the write queue.

    If the response is truncated (finish_reason='length'), the answer is
    left empty so a second pass with a larger max_length can retry it.
    Retries transient errors up to *max_retries* times.
    """
    async with semaphore:
        prompt = sanitize_messages(prompt)
        kwargs = dict(
            model=model,
            messages=prompt,
            max_tokens=max_length,
            temperature=temperature,
        )

        thinking, answer = "", ""
        truncated = False
        for attempt in range(1, max_retries + 1):
            t0 = time.monotonic()
            try:
                res = await client.chat.completions.create(**kwargs)
                choice = res.choices[0]
                message = choice.message
                content = message.content or ""
                # SGLang/vLLM separate thinking into reasoning_content for
                # thinking models (e.g. Qwen3.5, QwQ). Read it if available.
                reasoning_content = getattr(message, "reasoning_content", None) or ""
                finish_reason = choice.finish_reason
                elapsed = time.monotonic() - t0
                print(f"✓ index {idx}: {elapsed:.1f}s ({finish_reason})")
            except Exception as e:
                elapsed = time.monotonic() - t0
                print(f"Error for index {idx} (attempt {attempt}/{max_retries}, {elapsed:.1f}s): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                break

            thinking, answer = parse_thinking(content)
            # Prefer the framework-separated reasoning over inline parsing
            if reasoning_content:
                thinking = reasoning_content.strip()

            if finish_reason == "length":
                # Hit max_tokens — leave answer empty so a second pass can
                # retry with a larger max_length.
                truncated = True
                answer = ""
                break

            if answer:
                break

            if attempt < max_retries:
                print(f"Empty response for index {idx} (attempt {attempt}/{max_retries}), retrying…")

        if truncated:
            print(f"⚠️  index {idx}: truncated at max_tokens={max_length}")
        elif not answer:
            print(f"⚠️  index {idx}: answer still empty after {max_retries} attempt(s).")

        await queue.put({
            "index": idx,
            "thinking": thinking,
            "answer": answer,
        })


async def run_pass(label, indices, all_prompts, next_client, args, max_length,
                   queue, semaphore):
    """Run one generation pass over a list of indices."""
    print(f"🚀 {label}: Processing {len(indices)} prompts (max_length={max_length})...")
    tasks = [
        get_response(idx, all_prompts[idx], next_client(), args.model,
                     max_length, args.temperature, semaphore, queue,
                     args.max_retries)
        for idx in indices
    ]
    for f in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        await f

async def main(args):
    # 1. Setup clients
    custom_timeout = httpx.Timeout(7200.0)

    # When worker URLs are available, create one OpenAI client per worker
    # and round-robin across them.  This bypasses the SGLang router entirely,
    # avoiding its connection-pool bottleneck and circuit breaker.
    if args.worker_urls:
        base_urls = [f"{url.rstrip('/')}/v1" for url in args.worker_urls]
        print(f"📡 Bypassing router — sending requests directly to {len(base_urls)} workers")
    else:
        base_urls = [args.base_url]

    http_clients = []
    clients = []
    per_client_conns = max(args.concurrent // len(base_urls), 16)
    for url in base_urls:
        limits = httpx.Limits(
            max_connections=per_client_conns,
            max_keepalive_connections=per_client_conns,
        )
        hc = httpx.AsyncClient(limits=limits, timeout=custom_timeout)
        http_clients.append(hc)
        clients.append(AsyncOpenAI(base_url=url, api_key="EMPTY", http_client=hc))

    # Atomic counter for round-robin
    _rr_counter = 0
    def next_client():
        nonlocal _rr_counter
        c = clients[_rr_counter % len(clients)]
        _rr_counter += 1
        return c

    # 2. Load Dataset
    if os.path.exists(args.dataset_path):
        dataset = load_from_disk(args.dataset_path)
        if isinstance(dataset, DatasetDict):
            if args.split not in dataset:
                available_splits = ", ".join(dataset.keys())
                raise ValueError(
                    f"Split '{args.split}' not found in local dataset at {args.dataset_path}. "
                    f"Available splits: {available_splits}"
                )
            dataset = dataset[args.split]
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
                    existing_responses[data["index"]] = data.get("answer", "")
        print(f"✅ Resuming from checkpoint: Found {len(existing_responses)} saved items in responses.jsonl.")
    elif args.retry_existing and "answer" in dataset.column_names:
        existing_responses = {
            idx: answer
            for idx, answer in enumerate(dataset["answer"])
        }
        print(f"✅ Found {len(existing_responses)} saved items in the dataset answer column.")

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
        for hc in http_clients:
            await hc.aclose()
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
        for hc in http_clients:
            await hc.aclose()
        return

    # 5. Setup Async Queue and Concurrency
    queue = asyncio.Queue()
    semaphore = asyncio.Semaphore(args.concurrent)

    # Start the background writer task
    writer = asyncio.create_task(writer_task(queue, output_jsonl))

    # 6. First pass at max_length
    await run_pass("First pass", valid_indices, all_prompts, next_client,
                   args, args.max_length, queue, semaphore)

    # Flush first-pass writes before scanning the jsonl
    await queue.put(None)
    await writer

    # 7. Second pass on anything that came back empty (truncated) —
    # retry with the extended max length.
    empty_indices = []
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if not has_saved_response(data.get("answer", "")):
                    empty_indices.append(data["index"])
    # Deduplicate while preserving the most recent status
    empty_indices = sorted(set(empty_indices))

    if empty_indices:
        queue = asyncio.Queue()
        writer = asyncio.create_task(writer_task(queue, output_jsonl))
        await run_pass("Second pass", empty_indices, all_prompts, next_client,
                       args, args.extended_max_length, queue, semaphore)
        await queue.put(None)
        await writer

    # Close all http clients properly
    for hc in http_clients:
        await hc.aclose()

    print("✅ All requests completed and written to JSONL.")

    # 8. (Optional) Final Reconstruction into Hugging Face Dataset
    print(f"💾 Reconstructing and saving final dataset to {args.output_dir}")

    # Load everything back into ordered memory just once at the end
    if "answer" in dataset.column_names:
        final_answers = list(dataset["answer"])
    else:
        final_answers = [""] * len(dataset)
    if "thinking" in dataset.column_names:
        final_thinking = list(dataset["thinking"])
    else:
        final_thinking = [""] * len(dataset)

    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                final_answers[data["index"]] = data.get("answer", "")
                final_thinking[data["index"]] = data.get("thinking", "")

    if "answer" in dataset.column_names:
        dataset = dataset.remove_columns("answer")
    dataset = dataset.add_column("answer", final_answers)

    if "thinking" in dataset.column_names:
        dataset = dataset.remove_columns("thinking")
    dataset = dataset.add_column("thinking", final_thinking)

    if "generation_model" in dataset.column_names:
        dataset = dataset.remove_columns("generation_model")
    basename = os.path.basename(args.model)
    dataset = dataset.add_column("generation_model", [basename] * len(dataset))

    generation_meta = json.dumps({
        "temperature": args.temperature,
        "max_length": args.max_length,
        "extended_max_length": args.extended_max_length,
    })
    if "generation_meta" in dataset.column_names:
        dataset = dataset.remove_columns("generation_meta")
    dataset = dataset.add_column("generation_meta", [generation_meta] * len(dataset))

    dataset.save_to_disk(args.output_dir)

    with open(os.path.join(args.output_dir, "model_used.txt"), "w") as f:
        f.write(basename)

if __name__ == "__main__":
    # python generate.py --dataset-path=allenai/Dolci-Instruct-DPO --output-dir=./datasets/tmpbax3 --model=moonshotai/Kimi-K2.5-dmelikidze --concurrent=250 --base-url=http://172.28.33.32:8080/v1
    uvloop.install()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--prompt-column-name", type=str, default="prompt", help="Name of the column containing the prompt/messages")
    parser.add_argument("--remove-last-message", action="store_true", help="Whether to remove the last message from the conversation history, e.g. if you take it from a 'chosen' column")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=8192, help="Max generation tokens for the first pass")
    parser.add_argument("--extended-max-length", type=int, default=16384, help="Max generation tokens for the second pass (retries truncated first-pass responses)")
    parser.add_argument("--split", type=str, default="train")

    parser.add_argument("--concurrent", type=int, default=1000)
    parser.add_argument("--base-url", type=str, default="https://serving.swissai.cscs.ch/")
    parser.add_argument("--worker-urls", type=str, nargs="*", default=[], help="Direct worker URLs for /metrics polling (bypasses router)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--retry-existing", action="store_true", help="Retry samples whose saved response is empty in responses.jsonl or an existing response column")
    parser.add_argument("--max-retries", type=int, default=3, help="Number of times to retry a prompt when the returned answer is empty (default: 3)")
    args = parser.parse_args()

    asyncio.run(main(args))
