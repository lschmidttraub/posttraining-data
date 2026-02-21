"""
Lightweight script to send requests to a previously started OpenAI API server.
To start a server, see https://github.com/swiss-ai/model-launch.

Usage:
    python generate.py --dataset-path allenai/Dolci-Instruct-DPO --output-dir ./datasets/dolci_qwen3_8b --model Qwen/Qwen3-8B --base-url http://172.28.42.128:8080/v1
"""

import os
import json
import asyncio
import argparse

from openai import AsyncOpenAI
from dotenv import load_dotenv
from datasets import load_from_disk, load_dataset, DatasetDict
from tqdm.asyncio import tqdm_asyncio

load_dotenv()


async def main(args):
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key="EMPTY", 
    )

    if os.path.exists(args.dataset_path):
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset(args.dataset_path)
        if isinstance(dataset, DatasetDict):
            dataset = dataset["train"]

    # dataset = dataset.select(range(1000))

    # Get prompts (everything except the last message)
    dataset = dataset.map(lambda x: {"prompt": x["chosen"][:-1]})
    
    semaphore = asyncio.Semaphore(1000)

    async def get_response(prompt):
        async with semaphore:
            res = await client.chat.completions.create(
                model=args.model,
                messages=prompt,
                max_tokens=8192,
                logprobs=True,
                top_logprobs=1,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = res.choices[0].message.content
            
            # Extract and format logprobs into a serializable list of dicts
            raw_logprobs = res.choices[0].logprobs.content
            formatted_logprobs = [{"token": lp.token, "logprob": lp.logprob} for lp in raw_logprobs] if raw_logprobs else []
            
            return {"response": content, "logprobs": formatted_logprobs}

    tasks = [get_response(p) for p in dataset["prompt"]]
    results = await tqdm_asyncio.gather(*tasks, desc="Generating")

    # Unpack the results into separate lists
    responses = [r["response"] for r in results]
    logprobs = [r["logprobs"] for r in results]

    # Add both as new columns
    dataset = dataset.add_column("response", responses)
    dataset = dataset.add_column("logprobs", logprobs)

    output_dir = args.output_dir
    dataset.save_to_disk(output_dir)
    # saving the model name used for generation in a text file in the output_dir
    with open(os.path.join(output_dir, "model_used.txt"), "w") as f:
        f.write(args.model)

    # Save the first sample as JSON to the output_dir
    with open(
        os.path.join(output_dir, "first_sample.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(dataset[0], f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--base-url", type=str, default="https://api.swissai.cscs.ch/v1"
    )
    args = parser.parse_args()

    asyncio.run(main(args))