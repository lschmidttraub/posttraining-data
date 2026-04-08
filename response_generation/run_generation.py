import os
import sys
import json
import time
import argparse
import subprocess
import signal
import shlex
import urllib.request


def maybe_preprocess_dataset(args):
    if not args.preprocess:
        if not args.dataset:
            raise ValueError("--dataset is required when not using --preprocess")
        return args.dataset

    if not args.preprocess_category:
        raise ValueError("--preprocess-category is required when --preprocess is set")
    if not args.preprocessed_dataset_dir:
        raise ValueError("--preprocessed-dataset-dir is required when --preprocess is set")

    preprocess_inner_cmd = [
        "python", "-u", "-m", "preprocessing.run",
        "--category", args.preprocess_category,
        "--output-dir", args.preprocessed_dataset_dir,
        "--batch-size", str(args.preprocess_batch_size),
    ]
    if args.preprocess_num_proc is not None:
        preprocess_inner_cmd.extend(["--num-proc", str(args.preprocess_num_proc)])

    preprocess_cmd = [
        "srun",
        "--overlap",
        "--het-group=" + str(args.client_hetgroup),
        "--environment=./response_generation/env/alignment.toml",
        "--container-writable",
        f"--container-workdir={os.getcwd()}",
        "bash",
        "-lc",
        "unset SSL_CERT_FILE && " + " ".join(shlex.quote(arg) for arg in preprocess_inner_cmd),
    ]

    print(f"🚀 Preprocessing dataset: {' '.join(preprocess_cmd)}")
    subprocess.run(preprocess_cmd, check=True)
    print(f"✅ Preprocessed dataset saved to: {args.preprocessed_dataset_dir}")
    return args.preprocessed_dataset_dir


def main():
    parser = argparse.ArgumentParser(description="Orchestrate SGLang server and Generation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None, help="Dataset path to use for generation")
    parser.add_argument("--split", type=str, default="train", help="Split of the dataset to use")
    parser.add_argument("--prompt-column-name", type=str, default="prompt", help="Name of the column in the dataset that contains the prompts")
    parser.add_argument("--remove-last-message", action="store_true", help="Whether to remove the last message from the conversation history")

    parser.add_argument("--base-output-dir", type=str, default="./output")
    parser.add_argument("--logs-dir", type=str, default="./logs")
    parser.add_argument("--job-time", type=str, default="12:00:00")
    parser.add_argument("--account", type=str, default="infra01")

    parser.add_argument("--slurm-nodes", type=int, default=1)
    parser.add_argument("--client-hetgroup", type=int, default=1, help="Hetgroup to use for client submission")
    parser.add_argument("--server-hetgroup", type=int, default=1, help="Hetgroup to use for server submission")
    parser.add_argument("--workers", type=int, default=1, help="Number of sglang workers")
    parser.add_argument("--nodes-per-worker", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--disable-ocf", action="store_true", help="Disable OCF optimization")

    parser.add_argument("--enforce-eager", action="store_true", help="Disable compilation/CUDA graphs in vLLM")

    parser.add_argument("--framework", type=str, default="sglang", help="Serving framework (e.g., sglang, vllm)")
    parser.add_argument("--env", type=str, help="Optional environment name for job submission (e.g., vllm_qwen35)", required=False)
    parser.add_argument("--glm", action="store_true", help="Enable GLM-specific serving config (sglang_glm env, EAGLE speculative decoding, custom parsers)")
    parser.add_argument("--pre-launch-cmds", type=str, default=None, help="Commands to run before launching framework (e.g., 'pip install blobfile')")
    parser.add_argument("--base-url", type=str, help="Base URL for the model server (overrides auto-discovery)", required=False)

    parser.add_argument("--preprocess", action="store_true", help="Preprocess a category before generation")
    parser.add_argument("--preprocess-category", type=str, default=None, help="Category key from MAPPER_REGISTRY to preprocess (required with --preprocess)")
    parser.add_argument("--preprocessed-dataset-dir", type=str, default=None, help="Where to save/load the preprocessed dataset (required with --preprocess)")
    parser.add_argument("--preprocess-batch-size", type=int, default=1000, help="Batch size for preprocessing")
    parser.add_argument("--preprocess-num-proc", type=int, default=None, help="Optional number of preprocessing worker processes")


    args = parser.parse_args()
    dataset_path = maybe_preprocess_dataset(args)
    model_short = args.model.split("/")[-1]
    scratch = os.environ.get("SCRATCH", "/tmp")
    os.makedirs(args.logs_dir, exist_ok=True)
    server_proc = None
    job_id = os.environ.get("SLURM_JOB_ID")

    def cleanup():
        nonlocal server_proc
        if server_proc and server_proc.poll() is None:
            print("🛑 Stopping serving step...")
            os.killpg(server_proc.pid, signal.SIGTERM)
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print("⚠️ Serving step did not exit after SIGTERM, sending SIGKILL.")
                os.killpg(server_proc.pid, signal.SIGKILL)
                server_proc.wait()

    def handle_signal(signum, frame):
        cleanup()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if not args.base_url:
        submit_cmd = [
            "uv", "run",
            "python", f"{scratch}/model-launch/legacy/serving/run_job.py",
            "--slurm-nodes", str(args.slurm_nodes),
            "--hetgroup", str(args.server_hetgroup),
            "--slurm-time", args.job_time,
            "--serving-framework", args.framework,
            "--worker-port", "8080",
            "--slurm-account", args.account,
        ]
        if args.env:
            env_name = args.env
        elif args.glm:
            env_name = "sglang_glm"
        else:
            env_name = args.framework
        submit_cmd.extend([
            "--slurm-environment", f"{scratch}/model-launch/legacy/serving/envs/{env_name}.toml"
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
            fw_args = f"--model-path {args.model} --host 0.0.0.0 --port 8080 --served-model-name {args.model} --dp-size {args.dp_size} --tp-size {args.tp_size} --trust-remote-code --enable-metrics"
            if args.glm:
                fw_args += " --tool-call-parser glm47 --reasoning-parser glm45"
                fw_args += " --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
                fw_args += " --mem-fraction-static 0.85 --disable-cuda-graph"
        elif args.framework == "vllm":
            fw_args = f"--model {args.model} --host 0.0.0.0 --port 8080 --served-model-name {args.model} --data-parallel-size {args.dp_size} --tensor-parallel-size {args.tp_size} --trust-remote-code"
            if args.enforce_eager:
                fw_args += " --enforce-eager"
            if "mistral" in args.model.lower():
                fw_args += " --tokenizer_mode mistral --load_format mistral --config_format mistral"
        else:
            raise ValueError(f"Invalid framework: {args.framework}")

        submit_cmd.extend(["--framework-args", fw_args])

        pre_launch = args.pre_launch_cmds or ""
        # if args.nodes_per_worker > 1:
        #     # Workaround: on Slingshot/CXI, separate srun steps get isolated
        #     # network credentials, so NCCL's OFI/CXI provider fails. Force
        #     # NCCL to use TCP sockets for inter-node communication instead.
        #     nccl_fix = "export NCCL_NET=Socket"
        #     pre_launch = f"{nccl_fix}; {pre_launch}" if pre_launch else nccl_fix
        
        if pre_launch:
            submit_cmd.extend(["--pre-launch-cmds", pre_launch])

        print(f"🚀 Submitting: {' '.join(submit_cmd)}")
        server_proc = subprocess.Popen(
            submit_cmd,
            cwd=args.logs_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

        if not job_id:
            cleanup()
            print("❌ Missing outer SLURM job ID in environment.")
            sys.exit(1)

        print(f"✅ Using outer job ID: {job_id}")
    else:
        print("ℹ️ Using provided base URL; skipping serving launch.")
        
    submit_dir = os.environ.get("SLURM_SUBMIT_DIR", os.getcwd())
    server_log = os.path.join(submit_dir, "logs", job_id, "log.out") if job_id else None
    base_url = args.base_url
    worker_urls = []
    target_prefix = "Router URL: " if args.workers > 1 else "All worker URLs: "

    while not base_url:
        if server_proc and server_proc.poll() is not None:
            launch_output = ""
            if server_proc.stdout is not None:
                launch_output = server_proc.stdout.read()
            print("❌ Serving step exited before publishing a URL.")
            if launch_output:
                print("--- SERVING OUTPUT ---")
                print(launch_output)
            sys.exit(1)

        if server_log and os.path.exists(server_log):
            with open(server_log, "r") as f:
                for line in f:
                    if line.startswith(target_prefix):
                        raw = line.split(target_prefix)[1].strip()
                        base_url = f"{raw}/v1" if args.workers > 1 else f"{raw.rsplit(':', 1)[0]}:8080/v1"
                    if line.startswith("All worker URLs: "):
                        worker_urls = line.split("All worker URLs: ")[1].strip().split()
            if base_url:
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

    print(f"📊 Extracted worker URLs: {worker_urls}")
    output_dir = os.path.join(args.base_output_dir, model_short)
    generate_args = [
        "python", "-u", "response_generation/generate.py",
        "--dataset-path", dataset_path,
        "--prompt-column-name", args.prompt_column_name,
        "--split", args.split,
        "--output-dir", output_dir,
        "--model", args.model,
        "--base-url", base_url,
        "--retry-existing",
    ]
    if worker_urls:
        generate_args.extend(["--worker-urls"] + worker_urls)
    if args.remove_last_message:
        generate_args.append("--remove-last-message")
    gen_cmd = [
        "srun",
        "--overlap",
        "--het-group=" + str(args.client_hetgroup),
        "--environment=./response_generation/env/alignment.toml",
        "--container-writable",
        f"--container-workdir={os.getcwd()}",
        "bash",
        "-lc",
        "unset SSL_CERT_FILE && " + " ".join(shlex.quote(arg) for arg in generate_args),
    ]
    try:
        subprocess.run(gen_cmd, check=True)
    finally:
        cleanup()

if __name__ == "__main__":
    main()
