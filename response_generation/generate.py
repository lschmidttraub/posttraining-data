import os
import re
import json
import asyncio
import argparse
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


def upsert_column(dataset, name, values):
    """Replace a dataset column if it exists, otherwise add it."""
    if name in dataset.column_names:
        dataset = dataset.remove_columns(name)
    return dataset.add_column(name, values)


def drop_columns_if_present(dataset, names):
    existing = [name for name in names if name in dataset.column_names]
    if existing:
        dataset = dataset.remove_columns(existing)
    return dataset


def get_dataset_value(dataset, column_name, idx, default=None):
    if column_name not in dataset.column_names:
        return default
    return dataset[column_name][idx]

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


def serialize_stop_reason(stop_reason):
    """Normalize backend-specific stop reasons into a string column."""
    if stop_reason is None:
        return ""
    if isinstance(stop_reason, str):
        return stop_reason
    return json.dumps(stop_reason, ensure_ascii=False)


def extract_usage_metadata(response):
    """Extract token usage fields when the backend provides them."""
    usage = getattr(response, "usage", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "reasoning_tokens": getattr(completion_details, "reasoning_tokens", None),
        "cached_prompt_tokens": getattr(prompt_details, "cached_tokens", None),
    }


def merge_dict(defaults, value):
    merged = dict(defaults)
    if isinstance(value, dict):
        merged.update({k: v for k, v in value.items() if v is not None})
        return merged
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return merged
        if isinstance(parsed, dict):
            merged.update({k: v for k, v in parsed.items() if v is not None})
    return merged


def build_generation_meta(*, model, temperature, max_length, value=None):
    return merge_dict({
        "model": model,
        "temperature": temperature,
        "max_length": max_length,
    }, value)


def build_generation_info(
    *,
    value=None,
    finish_reason="",
    stop_reason="",
    generation_error="",
    prompt_tokens=None,
    completion_tokens=None,
    total_tokens=None,
    reasoning_tokens=None,
    cached_prompt_tokens=None,
):
    return merge_dict({
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "generation_error": generation_error,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
    }, value)

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
        finish_reason = ""
        stop_reason = ""
        usage_metadata = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "reasoning_tokens": None,
            "cached_prompt_tokens": None,
        }
        generation_error = ""
        try:
            # Sanitize messages to avoid Mistral tokenizer tool_calls issues
            prompt = sanitize_messages(prompt)
            
            kwargs = dict(
                model=model,
                messages=prompt,
                max_tokens=max_length,
                temperature=temperature,
            )
            
            res = await client.chat.completions.create(**kwargs)
            choice = res.choices[0]
            message = choice.message
            content = message.content or ""
            # SGLang/vLLM separate thinking into reasoning_content for
            # thinking models (e.g. Qwen3.5, QwQ). Read it if available.
            reasoning_content = getattr(message, "reasoning_content", None) or ""
            finish_reason = getattr(choice, "finish_reason", "") or ""
            stop_reason = serialize_stop_reason(getattr(choice, "stop_reason", None))
            usage_metadata = extract_usage_metadata(res)
        except Exception as e:
            print(f"Error for index {idx}: {e}")
            content = ""
            reasoning_content = ""
            generation_error = str(e)

        thinking, answer = parse_thinking(content)
        # Prefer the framework-separated reasoning over inline parsing
        if reasoning_content:
            thinking = reasoning_content.strip()
        await queue.put({
            "index": idx,
            "thinking": thinking,
            "answer": answer,
            "generation_meta": build_generation_meta(
                model=model,
                temperature=temperature,
                max_length=max_length,
            ),
            "generation_info": build_generation_info(
                finish_reason=finish_reason,
                stop_reason=stop_reason,
                generation_error=generation_error,
                **usage_metadata,
            ),
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
    tasks = [
        get_response(idx, all_prompts[idx], client, args.model, args.max_length, args.temperature, semaphore, queue) 
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
    if "answer" in dataset.column_names:
        final_answers = list(dataset["answer"])
    else:
        final_answers = [""] * len(dataset)
    if "thinking" in dataset.column_names:
        final_thinking = list(dataset["thinking"])
    else:
        final_thinking = [""] * len(dataset)
    final_generation_meta = [
        build_generation_meta(
            model=get_dataset_value(dataset, "generation_model", idx, args.model) or args.model,
            temperature=args.temperature,
            max_length=args.max_length,
            value=get_dataset_value(dataset, "generation_meta", idx),
        )
        for idx in range(len(dataset))
    ]
    final_generation_info = [
        build_generation_info(
            value=get_dataset_value(dataset, "generation_info", idx),
            finish_reason=get_dataset_value(dataset, "finish_reason", idx, "") or "",
            stop_reason=get_dataset_value(dataset, "stop_reason", idx, "") or "",
            generation_error=get_dataset_value(dataset, "generation_error", idx, "") or "",
            prompt_tokens=get_dataset_value(dataset, "prompt_tokens", idx),
            completion_tokens=get_dataset_value(dataset, "completion_tokens", idx),
            total_tokens=get_dataset_value(dataset, "total_tokens", idx),
            reasoning_tokens=get_dataset_value(dataset, "reasoning_tokens", idx),
            cached_prompt_tokens=get_dataset_value(dataset, "cached_prompt_tokens", idx),
        )
        for idx in range(len(dataset))
    ]

    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                final_answers[data["index"]] = data.get("answer", "")
                final_thinking[data["index"]] = data.get("thinking", "")
                final_generation_meta[data["index"]] = build_generation_meta(
                    model=args.model,
                    temperature=args.temperature,
                    max_length=args.max_length,
                    value=data.get("generation_meta"),
                )
                final_generation_info[data["index"]] = build_generation_info(
                    value=data.get("generation_info"),
                    finish_reason=data.get("finish_reason", "") or "",
                    stop_reason=data.get("stop_reason", "") or "",
                    generation_error=data.get("generation_error", "") or "",
                    prompt_tokens=data.get("prompt_tokens"),
                    completion_tokens=data.get("completion_tokens"),
                    total_tokens=data.get("total_tokens"),
                    reasoning_tokens=data.get("reasoning_tokens"),
                    cached_prompt_tokens=data.get("cached_prompt_tokens"),
                )

    dataset = upsert_column(dataset, "answer", final_answers)
    dataset = upsert_column(dataset, "thinking", final_thinking)
    dataset = upsert_column(dataset, "generation_meta", final_generation_meta)
    dataset = upsert_column(dataset, "generation_info", final_generation_info)
    dataset = drop_columns_if_present(dataset, [
        "finish_reason",
        "stop_reason",
        "generation_error",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cached_prompt_tokens",
        "generation_model",
    ])

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
    parser.add_argument("--max-length", type=int, default=8096)
    parser.add_argument("--split", type=str, default="train")
    
    # Increased default concurrency to better saturate the 16 nodes
    parser.add_argument("--concurrent", type=int, default=2000)
    parser.add_argument("--base-url", type=str, default="https://serving.swissai.cscs.ch/")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--retry-existing", action="store_true", help="Retry samples whose saved response is empty in responses.jsonl or an existing response column")
    args = parser.parse_args()

    asyncio.run(main(args))
