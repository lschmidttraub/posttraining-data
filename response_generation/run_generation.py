import os
import sys
import time
import argparse
import subprocess
import urllib.request
import urllib.error


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate SGLang server and Generation"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model path (e.g., Qwen/Qwen3-8B)"
    )
    parser.add_argument(
        "--dataset", type=str, default="allenai/Dolci-Instruct-DPO", help="Dataset path"
    )
    parser.add_argument(
        "--base-output-dir", type=str, default="./output", help="Base output directory"
    )
    parser.add_argument(
        "--job-time", type=str, default="09:00:00", help="Job time limit"
    )

    # New arguments exposed to the user
    parser.add_argument(
        "--slurm-nodes",
        type=int,
        default=1,
        help="Total number of nodes to allocate for the server",
    )
    parser.add_argument(
        "--dp-size", type=int, default=4, help="Data parallelism size (GPUs per node)"
    )
    parser.add_argument(
        "--tp-size", type=int, default=1, help="Tensor parallelism size (GPUs per node)"
    )

    args = parser.parse_args()

    model_full = args.model
    model_short = model_full.split("/")[-1]
    scratch = os.environ.get("SCRATCH", "/tmp")

    # 1. Launch the Server
    print(f"🚀 Submitting SGLang server job for {model_full}...")

    # Injecting the new dynamic arguments
    submit_cmd = [
        "python",
        f"{scratch}/model-launch/serving/submit_job.py",
        "--slurm-nodes",
        str(args.slurm_nodes),
        "--slurm-time",
        str(args.job_time),
        "--serving-framework",
        "sglang",
        "--slurm-environment",
        f"{scratch}/model-launch/serving/sglang.toml",
        "--framework-args",
        f"--model-path {model_full} --host 0.0.0.0 --port 8080 --served-model-name {model_full} --max-running-requests 2000 --mem-fraction-static 0.95 --dp-size {args.dp_size} --tp-size {args.tp_size}",
    ]

    job_id = None
    try:
        result = subprocess.run(submit_cmd, capture_output=True, text=True, check=True)
        for line in result.stderr.splitlines() + result.stdout.splitlines():
            if "Job submitted successfully with ID:" in line:
                job_id = line.split()[-1]
                break

        if job_id:
            print(f"✅ Server submitted successfully! Slurm Job ID: {job_id}")
        else:
            print("❌ Error: Could not parse Slurm Job ID.")
            sys.exit(1)

    except subprocess.CalledProcessError as e:
        print(f"❌ Error submitting job:\n{e.stderr}\n{e.stdout}")
        sys.exit(1)

    # 2. Parse the Log File for the IP
    log_file = f"./logs/{job_id}/log.out"
    print(f"⏳ Waiting for SLURM log file at: {log_file}")

    base_url = None
    while True:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                content = f.read()
                if "All worker URLs: " in content:
                    for line in content.splitlines():
                        if line.startswith("All worker URLs: "):
                            raw_url = line.split("All worker URLs: ")[1].strip()
                            ip_part = raw_url.rsplit(":", 1)[0]
                            base_url = f"{ip_part}:8080/v1"
                            break
            if base_url:
                break
        time.sleep(5)

    print(f"🔗 Extracted Base URL: {base_url}")

    # 3. Wait for SGLang to finish loading weights
    health_url = base_url.replace("/v1", "/health")
    print(
        f"⏳ Waiting for SGLang to load model weights and bind to port (checking {health_url})..."
    )

    while True:
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.getcode() == 200:
                    break
        except urllib.error.URLError:
            pass
        time.sleep(10)

    print("✅ Server is fully initialized and ready!")

    output_dir = f"{args.base_output_dir}/{model_short}"

    print(f"⚙️ Launching generate.py. Outputting to {output_dir}")
    generate_cmd = [
        "python",
        "generate.py",
        "--dataset-path",
        args.dataset,
        "--output-dir",
        output_dir,
        "--model",
        model_full,
        "--base-url",
        base_url,
    ]

    try:
        subprocess.run(generate_cmd, check=True)
        print("🎉 Generation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Generation script failed with error: {e}")
    finally:
        # 5. Cleanup
        print(f"🧹 Cleaning up SGLang server (scancel {job_id})...")
        subprocess.run(["scancel", job_id])
        print("Done.")


if __name__ == "__main__":
    main()
