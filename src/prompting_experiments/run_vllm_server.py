#!/usr/bin/env python3
"""
Script to launch vLLM server for chemistry prompting experiments.
"""

import argparse
import subprocess
import sys
import os
import signal
import time

def launch_vllm_server(
    model_name: str = "meta-llama/Llama-3.2-1B",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
    host: str = "0.0.0.0"
):
    """Launch vLLM server with specified configuration."""
    
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(port),
        "--host", host,
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--served-model-name", model_name.split("/")[-1]  # Use just the model name without org
    ]
    
    print(f"Launching vLLM server with command:")
    print(" ".join(cmd))
    print(f"\nServer will be available at: http://{host}:{port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        process = subprocess.Popen(cmd)
        process.wait()
    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Stopping server...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("Force killing server...")
            process.kill()
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Launch vLLM server for chemistry experiments")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B",
                        help="Model name to serve")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to serve on")
    parser.add_argument("--tensor_parallel_size", type=int, default=4,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization fraction")
    parser.add_argument("--max_model_len", type=int, default=16384,
                        help="Maximum model context length")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind to")
    
    args = parser.parse_args()
    
    launch_vllm_server(
        model_name=args.model,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        host=args.host
    )

if __name__ == "__main__":
    main()
