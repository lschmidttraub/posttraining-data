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


class AdaptiveSemaphore:
    """AIMD-style concurrency controller.

    On success: concurrency += additive_increase  (up to max_concurrency)
    On overload (503): concurrency *= multiplicative_decrease, plus backoff
    """

    def __init__(self, max_concurrency, min_concurrency=4,
                 additive_increase=1, multiplicative_decrease=0.5,
                 cooldown_seconds=5.0):
        self.max_concurrency = max_concurrency
        self.min_concurrency = min_concurrency
        self.additive_increase = additive_increase
        self.multiplicative_decrease = multiplicative_decrease
        self.cooldown_seconds = cooldown_seconds
        self._limit = max_concurrency
        self._in_flight = 0
        self._last_reduction = 0.0  # monotonic timestamp of last reduction
        self._lock = asyncio.Lock()
        self._can_proceed = asyncio.Event()
        self._can_proceed.set()

    @property
    def current_limit(self):
        return int(self._limit)

    async def acquire(self):
        while True:
            async with self._lock:
                if self._in_flight < int(self._limit):
                    self._in_flight += 1
                    return
            # Wait until a slot opens (released or limit raised)
            self._can_proceed.clear()
            await self._can_proceed.wait()

    async def release(self, success=True):
        async with self._lock:
            self._in_flight -= 1
            if success:
                self._limit = min(self._limit + self.additive_increase,
                                  self.max_concurrency)
            # Always wake up waiters on release
        self._can_proceed.set()

    async def on_overload(self):
        async with self._lock:
            now = time.monotonic()
            if now - self._last_reduction < self.cooldown_seconds:
                return  # already reduced recently, ignore this 503
            new_limit = max(self._limit * self.multiplicative_decrease,
                            self.min_concurrency)
            if new_limit < self._limit:
                print(f"⚡ Overload detected — reducing concurrency "
                      f"{int(self._limit)} → {int(new_limit)}")
                self._limit = new_limit
                self._last_reduction = now

    async def adjust_from_queue_depth(self, waiting, running):
        """Proactively adjust concurrency based on server queue depth.

        Called by the queue-depth monitor before 503s happen.
        Uses the *trend* (whether the queue is growing) rather than the
        absolute size, since a large queue at startup is expected.
        """
        async with self._lock:
            prev = getattr(self, "_prev_waiting", None)
            self._prev_waiting = waiting

            if prev is None:
                return  # first sample — no trend yet, skip

            queue_growing = waiting > prev
            queue_shrinking = waiting < prev * 0.9  # 10%+ decrease

            if queue_growing and waiting > running:
                # Queue is actively getting worse — reduce gently
                new_limit = max(self._limit * 0.9, self.min_concurrency)
                if new_limit < self._limit:
                    print(f"📊 Queue growing (waiting={int(waiting)}, "
                          f"running={int(running)}) — reducing concurrency "
                          f"{int(self._limit)} → {int(new_limit)}")
                    self._limit = new_limit
            elif queue_shrinking and waiting == 0:
                # Queue fully drained — ramp up
                new_limit = min(self._limit * 1.05 + 1, self.max_concurrency)
                self._limit = new_limit
        self._can_proceed.set()

def _parse_prometheus_gauge(text, metric_name):
    """Extract a gauge value from Prometheus-format metrics text.

    Matches lines like 'sglang:num_queue_reqs{labels...} 2452.0'.
    Uses the metric name portion (before '{' or ' ') to avoid partial matches
    (e.g. 'num_queue_reqs' should not match 'num_grammar_queue_reqs').
    """
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        # Extract the metric name (everything before '{' or the first space)
        key = line.split("{")[0].split()[0] if line else ""
        if key.endswith(metric_name):
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    return float(parts[-1])
                except ValueError:
                    pass
    return None


def _extract_metrics(text):
    """Extract waiting/running request counts from Prometheus metrics text.

    Handles both SGLang and vLLM metric names.
    """
    # SGLang: num_queue_reqs (waiting) + num_running_reqs
    waiting = _parse_prometheus_gauge(text, "num_queue_reqs")
    running = _parse_prometheus_gauge(text, "num_running_reqs")
    # vLLM fallback
    if waiting is None:
        waiting = _parse_prometheus_gauge(text, "num_requests_waiting")
    if running is None:
        running = _parse_prometheus_gauge(text, "num_requests_running")
    return waiting, running


async def queue_depth_monitor(base_url, worker_urls, semaphore,
                              poll_interval=2.0):
    """Background task that polls /metrics and proactively adjusts concurrency.

    Uses its own httpx client (separate connection pool) so metrics polling
    is never starved by in-flight inference requests.

    When worker_urls are provided (multi-worker setup behind a router), polls
    each worker directly and aggregates.  Otherwise falls back to the base_url.
    If /metrics is unavailable, backs off and retries indefinitely rather than
    giving up permanently.
    """
    # Build the list of metrics URLs to poll
    if worker_urls:
        metrics_urls = [f"{url.rstrip('/')}/metrics" for url in worker_urls]
    else:
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        metrics_urls = [f"{root}/metrics"]

    print(f"📊 Queue monitor: polling {metrics_urls}")

    # Dedicated lightweight client — never contends with inference requests
    metrics_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=4, max_keepalive_connections=4),
        timeout=httpx.Timeout(10.0),
    )

    current_interval = poll_interval
    try:
        while True:
            try:
                total_waiting, total_running = 0.0, 0.0
                any_success = False

                responses = await asyncio.gather(
                    *(metrics_client.get(url) for url in metrics_urls),
                    return_exceptions=True,
                )
                for url, resp in zip(metrics_urls, responses):
                    if isinstance(resp, Exception):
                        print(f"📊 Queue monitor: {url} error: "
                              f"{type(resp).__name__}: {resp}")
                        continue
                    if resp.status_code != 200:
                        print(f"📊 Queue monitor: {url} returned {resp.status_code}")
                        continue
                    waiting, running = _extract_metrics(resp.text)
                    if waiting is not None and running is not None:
                        total_waiting += waiting
                        total_running += running
                        any_success = True
                    else:
                        print(f"📊 Queue monitor: {url} returned 200 but "
                              f"parse failed (waiting={waiting}, running={running})")

                if any_success:
                    current_interval = poll_interval  # reset backoff
                    await semaphore.adjust_from_queue_depth(
                        total_waiting, total_running)
                else:
                    # Back off but keep retrying — the server may recover
                    current_interval = min(current_interval * 1.5, 30.0)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"📊 Queue monitor: unexpected error: "
                      f"{type(e).__name__}: {e}")
                current_interval = min(current_interval * 1.5, 30.0)

            await asyncio.sleep(current_interval)
    except asyncio.CancelledError:
        pass
    finally:
        await metrics_client.aclose()


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

def _is_overload_error(exc):
    """Return True when the exception signals server overload (503)."""
    msg = str(exc)
    return "503" in msg or "no_available_workers" in msg


async def get_response(idx, prompt, client, model, max_length, temperature, semaphore, queue, max_retries=3):
    """Fetches the response and immediately puts it in the write queue.

    Retries up to *max_retries* times when the returned answer is empty.
    Distinguishes two failure modes:
      1. Thinking exhausted the token budget (thinking present, answer empty)
         → doubles max_tokens on the next attempt.
      2. Completely empty response (both thinking and answer empty), typically
         a transient server error → retries with the same parameters.

    On 503 / overload errors the adaptive semaphore is notified so it can
    reduce concurrency, and an exponential backoff is applied before retry.
    """
    await semaphore.acquire()
    success = True
    try:
        prompt = sanitize_messages(prompt)
        current_max_tokens = max_length
        kwargs = dict(
            model=model,
            messages=prompt,
            max_tokens=current_max_tokens,
            temperature=temperature,
        )

        thinking, answer = "", ""
        backoff = 2.0  # initial backoff seconds for overload retries
        for attempt in range(1, max_retries + 1):
            try:
                res = await client.chat.completions.create(**kwargs)
                message = res.choices[0].message
                content = message.content or ""
                # SGLang/vLLM separate thinking into reasoning_content for
                # thinking models (e.g. Qwen3.5, QwQ). Read it if available.
                reasoning_content = getattr(message, "reasoning_content", None) or ""
            except Exception as e:
                print(f"Error for index {idx} (attempt {attempt}/{max_retries}): {e}")
                content = ""
                reasoning_content = ""

                if _is_overload_error(e):
                    success = False
                    await semaphore.on_overload()
                    if attempt < max_retries:
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        # Re-acquire after backoff (respects new lower limit)
                        await semaphore.release(success=False)
                        await semaphore.acquire()
                        success = True  # reset for this new attempt
                        print(f"Empty response for index {idx} (attempt {attempt}/{max_retries}), retrying…")
                        continue

            thinking, answer = parse_thinking(content)
            # Prefer the framework-separated reasoning over inline parsing
            if reasoning_content:
                thinking = reasoning_content.strip()

            if answer:
                break

            if attempt < max_retries:
                if thinking:
                    # Thinking exhausted the token budget — double max_tokens
                    # so the model has room for an answer on the next attempt.
                    current_max_tokens *= 2
                    kwargs["max_tokens"] = current_max_tokens
                    print(f"Token budget exhausted for index {idx} (attempt {attempt}/{max_retries}), "
                          f"doubling max_tokens to {current_max_tokens} and retrying…")
                else:
                    # Completely empty response (transient error) — retry as-is.
                    print(f"Empty response for index {idx} (attempt {attempt}/{max_retries}), retrying…")

        if not answer:
            print(f"⚠️  index {idx}: answer still empty after {max_retries} attempt(s).")

        await queue.put({
            "index": idx,
            "thinking": thinking,
            "answer": answer,
        })
    finally:
        await semaphore.release(success=success)

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
    semaphore = AdaptiveSemaphore(args.concurrent)
    
    # Start the background writer task
    writer = asyncio.create_task(writer_task(queue, output_jsonl))

    # Preflight: verify /metrics connectivity before starting the monitor
    print(f"📊 Worker URLs received: {args.worker_urls}")
    if args.worker_urls:
        for url in args.worker_urls:
            metrics_url = f"{url.rstrip('/')}/metrics"
            try:
                resp = await http_client.get(metrics_url, timeout=5.0)
                sample = resp.text[:200] if resp.status_code == 200 else ""
                print(f"📊 Preflight {metrics_url}: status={resp.status_code} "
                      f"body_preview={sample!r}")
            except Exception as e:
                print(f"📊 Preflight {metrics_url}: FAILED ({type(e).__name__}: {e})")

    # Start the queue-depth monitor (proactive concurrency adjustment)
    # Uses its own httpx client internally — no shared connection pool
    monitor = asyncio.create_task(
        queue_depth_monitor(args.base_url, args.worker_urls, semaphore))

    # 6. Create and Run Tasks
    print(f"🚀 Processing {len(valid_indices)} prompts...")
    tasks = [
        get_response(idx, all_prompts[idx], client, args.model, args.max_length, args.temperature, semaphore, queue, args.max_retries)
        for idx in valid_indices
    ]
    
    # Use as_completed to avoid gathering huge arrays of objects into memory
    for f in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        await f

    # 7. Shutdown monitor and writer gracefully
    monitor.cancel()
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
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--split", type=str, default="train")
    
    # Increased default concurrency to better saturate the 16 nodes
    parser.add_argument("--concurrent", type=int, default=1000)
    parser.add_argument("--base-url", type=str, default="https://serving.swissai.cscs.ch/")
    parser.add_argument("--worker-urls", type=str, nargs="*", default=[], help="Direct worker URLs for /metrics polling (bypasses router)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--retry-existing", action="store_true", help="Retry samples whose saved response is empty in responses.jsonl or an existing response column")
    parser.add_argument("--max-retries", type=int, default=3, help="Number of times to retry a prompt when the returned answer is empty (default: 3)")
    args = parser.parse_args()

    asyncio.run(main(args))
