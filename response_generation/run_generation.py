import os
import sys
import json
import time
import argparse
import subprocess
import urllib.request

def main():
    parser = argparse.ArgumentParser(description="Orchestrate SGLang server and Generation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="allenai/Dolci-Instruct-DPO")
    parser.add_argument("--prompt-column-name", type=str, default="prompt", help="Name of the column in the dataset that contains the prompts")
    parser.add_argument("--remove-last-message", action="store_true", help="Whether to remove the last message from the conversation history")
    parser.add_argument("--base-output-dir", type=str, default="./output")
    parser.add_argument("--logs-dir", type=str, default="./logs")
    parser.add_argument("--job-time", type=str, default="12:00:00")

    parser.add_argument("--account", type=str, default="infra01")

    parser.add_argument("--slurm-nodes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1, help="Number of sglang workers")
    parser.add_argument("--nodes-per-worker", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--disable-ocf", action="store_true", help="Disable OCF optimization")
    parser.add_argument("--framework", type=str, default="sglang", help="Serving framework (e.g., sglang, vllm)")
    parser.add_argument("--no-reasoning-kwargs", action="store_true", help="Disable passing chat_template_kwargs for reasoning")
    parser.add_argument("--env", type=str, help="Optional environment name for job submission (e.g., vllm_qwen35)", required=False)
    parser.add_argument("--split", type=str, default="train", help="Split of the dataset to use")
    parser.add_argument("--base-url", type=str, help="Base URL for the model server (overrides auto-discovery)", required=False)


    args = parser.parse_args()
    model_short = args.model.split("/")[-1]
    scratch = os.environ.get("SCRATCH", "/tmp")
    os.makedirs(args.logs_dir, exist_ok=True)
    if not args.base_url:
        submit_cmd = [
            "python", f"{scratch}/model-launch/legacy/serving/submit_job.py",
            "--slurm-nodes", str(args.slurm_nodes),
            "--slurm-time", args.job_time,
            "--serving-framework", args.framework,
            "--worker-port", "8080",
            "--slurm-account", args.account,
        ]
        if args.env:
            submit_cmd.extend([
                "--slurm-environment", f"{scratch}/model-launch/legacy/serving/envs/{args.env}.toml"
            ])
        else:
            submit_cmd.extend([
                "--slurm-environment", f"{scratch}/model-launch/legacy/serving/envs/{args.framework}.toml"
            ])
            
        if args.workers > 1:
            submit_cmd.extend([
                "--workers", str(args.workers),
                "--nodes-per-worker", str(args.nodes_per_worker),
                "--use-router",
                "--router-environment", f"{scratch}/model-launch/legacy/serving/envs/sglang.toml",
            ])

        if args.disable_ocf:
            submit_cmd.append("--disable-ocf")
        
        if args.framework == "sglang":
            fw_args = f"--model-path {args.model} --host 0.0.0.0 --port 8080 --served-model-name {args.model} --dp-size {args.dp_size} --tp-size {args.tp_size} --trust-remote-code"
        elif args.framework == "vllm":
            fw_args = f"--model {args.model} --host 0.0.0.0 --port 8080 --served-model-name {args.model} --data-parallel-size {args.dp_size} --tensor-parallel-size {args.tp_size} --trust-remote-code"
            if "mistral" in args.model.lower():
                fw_args += " --tokenizer_mode mistral --load_format mistral --config_format mistral"
        else:
            raise ValueError(f"Invalid framework: {args.framework}")
        
        submit_cmd.extend(["--framework-args", fw_args])

        print(f"🚀 Submitting: {' '.join(submit_cmd)}")
        result = subprocess.run(submit_cmd, cwd=args.logs_dir, capture_output=True, text=True, check=True)
        combined_output = result.stdout + "\n" + result.stderr
        job_id = None
        
        for line in combined_output.splitlines():
            if "Job submitted successfully with ID:" in line:
                job_id = line.split()[-1].strip()
                break
                
        if not job_id:
            print("❌ Failed to parse Job ID from output.")
            print("--- STDOUT ---")
            print(result.stdout)
            print("--- STDERR ---")
            print(result.stderr)
            sys.exit(1)

        print(f"✅ Found Job ID: {job_id}")
    else:
        # TODO: bad because we can't cancel the running server this way.
        job_id = ""
        
    log_file = f"{args.logs_dir}/logs/{job_id}/log.out"
    base_url = args.base_url
    target_prefix = "Router URL: " if args.workers > 1 else "All worker URLs: "

    while not base_url:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                content = f.read()
                if target_prefix in content:
                    for line in content.splitlines():
                        if line.startswith(target_prefix):
                            raw = line.split(target_prefix)[1].strip()
                            base_url = f"{raw}/v1" if args.workers > 1 else f"{raw.rsplit(':', 1)[0]}:8080/v1"
                            break
        time.sleep(5)

    health_url = base_url.replace("/v1", "/health")
    completions_url = f"{base_url}/chat/completions"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    print("⏳ Waiting for server health check...")
    while True:
        try:
            with opener.open(urllib.request.Request(health_url), timeout=5) as resp:
                if resp.getcode() == 200: break
        except: pass
        time.sleep(10)
    print("✅ Server health check passed.")

    # Send a real inference request to verify the full pipeline works.
    # This catches router race conditions (workers not yet registered) and
    # routing issues (e.g. SGLang router circuit breakers with vLLM backends).
    probe_payload = json.dumps({
        "model": args.model,
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 4,
        "temperature": 0,
    }).encode()
    probe_headers = {"Content-Type": "application/json"}

    print("⏳ Sending readiness probe (test inference request)...")
    probe_delay = 5
    while True:
        try:
            req = urllib.request.Request(completions_url, data=probe_payload, headers=probe_headers)
            with opener.open(req, timeout=120) as resp:
                if resp.getcode() == 200:
                    print("✅ Readiness probe succeeded — server is fully operational.")
                    break
        except urllib.error.HTTPError as e:
            print(f"  Readiness probe returned HTTP {e.code}, retrying in {probe_delay}s...")
        except Exception as e:
            print(f"  Readiness probe failed ({e}), retrying in {probe_delay}s...")
        time.sleep(probe_delay)
        probe_delay = min(probe_delay * 2, 60)

    output_dir = os.path.join(args.base_output_dir, model_short)
    gen_cmd = [
        "python", "response_generation/generate.py",
        "--dataset-path", args.dataset,
        "--prompt-column-name", args.prompt_column_name,
        "--split", args.split,
        "--output-dir", output_dir,
        "--model", args.model,
        "--base-url", base_url,
        "--retry-existing",
    ]
    if args.remove_last_message:
        gen_cmd.append("--remove-last-message")
    if args.no_reasoning_kwargs:
        gen_cmd.append("--no-reasoning-kwargs")
    subprocess.run(gen_cmd, check=True)
    
    subprocess.run(["scancel", job_id])

if __name__ == "__main__":
    main()
