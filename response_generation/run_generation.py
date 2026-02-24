import os
import sys
import time
import argparse
import subprocess
import urllib.request
import urllib.error

def main():
    parser = argparse.ArgumentParser(description="Orchestrate SGLang server and Generation")
    parser.add_argument("--model", type=str, required=True, help="Model path (e.g., Qwen/Qwen3-8B)")
    parser.add_argument("--dataset", type=str, default="allenai/Dolci-Instruct-DPO", help="Dataset path")
    
    # New arguments exposed to the user
    parser.add_argument("--slurm-nodes", type=int, default=1, help="Total number of nodes to allocate for the server")
    parser.add_argument("--dp-size", type=int, default=4, help="Data parallelism size (GPUs per node)")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallelism size (GPUs per node)")
    
    args = parser.parse_args()

    model_full = args.model
    model_short = model_full.split("/")[-1]
    scratch = os.environ.get("SCRATCH", "/tmp")
    
    # 1. Launch the Server
    print(f"🚀 Submitting SGLang server job for {model_full}...")
    
    # Base command
    submit_cmd = [
        "python", f"{scratch}/model-launch/serving/submit_job.py",
        "--slurm-nodes", str(args.slurm_nodes),
        "--slurm-time", "09:00:00",
        "--serving-framework", "sglang",
        "--worker-port", "8000",
        "--slurm-environment", f"{scratch}/model-launch/serving/sglang_latest.toml",
    ]
    
    # --- MODIFICATION 1: Inject router flags for multi-node ---
    if args.slurm_nodes > 1:
        submit_cmd.extend([
            "--workers", str(args.slurm_nodes),
            "--nodes-per-worker", "1",
            "--use-router"
        ])
    
    # Add the framework args
    submit_cmd.extend([
        "--framework-args", f"--model-path {model_full} --host 0.0.0.0 --port 8080 --served-model-name {model_full} --dp-size {args.dp_size} --tp-size {args.tp_size}"
    ])
    
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
    
    # --- MODIFICATION 2: Look for the correct log prefix based on architecture ---
    target_log_prefix = "Router URL: " if args.slurm_nodes > 1 else "All worker URLs: "
    
    while True:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                content = f.read()
                if target_log_prefix in content:
                    for line in content.splitlines():
                        if line.startswith(target_log_prefix):
                            raw_url = line.split(target_log_prefix)[1].strip()
                            
                            if args.slurm_nodes > 1:
                                # Router URL already has the correct port (e.g., http://IP:30000)
                                base_url = f"{raw_url}/v1"
                            else:
                                # For single worker, use the existing logic to force port 8080
                                ip_part = raw_url.rsplit(":", 1)[0]
                                base_url = f"{ip_part}:8080/v1"
                            break
            if base_url:
                break
        time.sleep(5) 
        
    print(f"🔗 Extracted Base URL: {base_url}")
    
    # 3. Wait for SGLang to finish loading weights
    health_url = base_url.replace("/v1", "/health")
    print(f"⏳ Waiting for SGLang to load model weights and bind to port (checking {health_url})...")
    
    # --- MODIFICATION 3: Bypass Python's global proxy for the health check ---
    proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)

    while True:
        try:
            req = urllib.request.Request(health_url)
            # Use the proxy-bypassing opener instead of standard urlopen
            with opener.open(req, timeout=5) as response:
                if response.getcode() == 200:
                    break
        except urllib.error.URLError:
            pass 
        time.sleep(10)
        
    print("✅ Server is fully initialized and ready!")

    output_dir = f"./output/{model_short}"
    
    print(f"⚙️ Launching generate.py. Outputting to {output_dir}")
    generate_cmd = [
        "python", "generate.py",
        "--dataset-path", args.dataset,
        "--output-dir", output_dir,
        "--model", model_full,
        "--base-url", base_url
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