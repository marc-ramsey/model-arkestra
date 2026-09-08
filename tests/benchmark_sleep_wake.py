#!/usr/bin/env python3
"""Benchmark llama.cpp sleep-wake vs cold start.

Measures time from request → first token + full response for:
1. Warm (already running, loaded)
2. Hot (sleeping, wake on request)
3. Cold (fresh process startup)
"""
import asyncio
import aiohttp
import subprocess
import time
import json
import sys
import os

MODEL_1B = (
    "/home/marc/.cache/huggingface/hub/models--bartowski--Llama-3.2-1B-Instruct-GGUF"
    "/blobs/6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83"
)

# If we have the 1.7B model available too:
MODEL_1_7B = (
    "/home/marc/.cache/huggingface/hub/models--bartowski--SmolLM2-1.7B-Instruct-GGUF"
    "/blobs/77665ea4815999596525c636fbeb56ba8b080b46ae85efef4f0d986a139834d7"
)

LLAMA_BIN = "/home/marc/local/llama.cpp/build-vulkan-radv/bin/llama-server"

PROMPT = "Say hello in 3 words."
SYSTEM_PROMPT = "Be brief and friendly."


async def health_ready(url: str, port: int, timeout=60) -> bool:
    """Wait for server to be ready."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
                    if resp.status == 200:
                        return True
        except Exception:
            pass
        await asyncio.sleep(0.3)
    return False


async def chat(url: str, model_path: str, prompt: str = PROMPT) -> tuple[float, float, dict]:
    """Send a chat request. Returns (first_token_ms, total_ms, response_data)."""
    payload = {
        "model": "",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 30,
        "temperature": 0.7,
    }

    start = time.monotonic()
    first_token_time = None
    
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{url}/v1/chat/completions", json=payload, timeout=60) as resp:
            data = await resp.json()
    
    total_ms = (time.monotonic() - start) * 1000
    
    # For streaming we'd need to parse SSE; for now just use non-streaming
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "<empty>")
    
    return total_ms, total_ms if first_token_time is None else first_token_time, data


async def get_health_status(url: str, port: int) -> dict:
    """Get health response."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
                return await resp.json()
    except Exception:
        return {"error": "unreachable"}


def start_server(model: str, port: int, sleep_idle: int = -1, log_prefix: str = "") -> subprocess.Popen:
    """Start llama-server and return Popen handle."""
    cmd = [
        LLAMA_BIN,
        "-m", model,
        "--ctx-size", "2048",
        "-ngl", "999",
        "--threads", "8",
        "--port", str(port),
        "--temp", "0.7",
    ]
    if sleep_idle > 0:
        cmd.extend(["--sleep-idle-seconds", str(sleep_idle)])
    
    print(f"[{log_prefix}] Starting: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def stop_server(proc: subprocess.Popen):
    """Gracefully stop a server."""
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


async def run_benchmark(model_name: str, model_path: str, port: int = 19000, repeats: int = 3):
    """Run the full benchmark for a single model."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    file_size = os.path.getsize(model_path) / (1024**2)
    print(f"GGUF size: {file_size:.0f} MB")
    print(f"Port: {port}")
    print(f"{'='*60}")

    results = {
        "warm": [],  # Already loaded
        "hot": [],   # Sleeping → wake on request
        "cold": [],  # Fresh process start
    }

    # ── Warm test ──────────────────────────────────────────────────────
    print(f"\n--- WARM (already loaded, no idle) ---")
    proc = start_server(model_path, port, sleep_idle=-1, log_prefix="WARM")
    
    if await health_ready("http://127.0.0.1", port):
        for i in range(repeats):
            total_ms, first_ms, data = await chat(f"http://127.0.0.1:{port}", model_path)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")[:50]
            results["warm"].append(total_ms)
            print(f"  Warm #{i+1}: {total_ms:.0f}ms → {content}")
    stop_server(proc)

    # ── Hot test (sleep → wake) ────────────────────────────────────────
    print(f"\n--- HOT (sleep-idle-seconds=5, then wake) ---")
    proc = start_server(model_path, port, sleep_idle=3, log_prefix="HOT")
    
    if await health_ready("http://127.0.0.1", port):
        # Fire one request to make it loaded
        print("  Loading model...")
        await chat(f"http://127.0.0.1:{port}", model_path)
        
        # Wait for idle sleep
        print("  Waiting for sleep (5s + margin)...")
        await asyncio.sleep(8)
        
        # Check health/status
        status = await get_health_status("http://127.0.0.1", port)
        print(f"  Health after idle: {json.dumps(status)}")
        
        for i in range(repeats):
            total_ms, first_ms, data = await chat(f"http://127.0.0.1:{port}", model_path)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")[:50]
            results["hot"].append(total_ms)
            print(f"  Hot  #{i+1}: {total_ms:.0f}ms → {content}")
            
            # Sleep again to test repeat wakes
            await asyncio.sleep(8)
    stop_server(proc)

    # ── Cold test ──────────────────────────────────────────────────────
    print(f"\n--- COLD (fresh process start) ---")
    proc = start_server(model_path, port, log_prefix="COLD")
    
    if await health_ready("http://127.0.0.1", port):
        for i in range(repeats):
            total_ms, first_ms, data = await chat(f"http://127.0.0.1:{port}", model_path)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")[:50]
            results["cold"].append(total_ms)
            print(f"  Cold #{i+1}: {total_ms:.0f}ms → {content}")
    stop_server(proc)

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY: {model_name}")
    print(f"{'State':<10} {'Min':>8} {'Avg':>8} {'Max':>8} {'Median':>8}")
    for state, vals in results.items():
        if not vals:
            continue
        import statistics
        avg = statistics.mean(vals)
        med = statistics.median(vals)
        mn = min(vals)
        mx = max(vals)
        print(f"{state:<10} {mn:>7.0f}ms {avg:>7.0f}ms {mx:>7.0f}ms {med:>7.0f}ms")

    # Compare
    hot_avg = statistics.mean(results["hot"]) if results["hot"] else 0
    cold_avg = statistics.mean(results["cold"]) if results["cold"] else 0
    warm_avg = statistics.mean(results["warm"]) if results["warm"] else 0
    
    if hot_avg > 0 and cold_avg > 0:
        saving_ms = hot_avg - cold_avg
        saving_pct = (saving_ms / hot_avg) * 100
        print(f"\nHot vs Cold: Hot saves ~{saving_ms:.0f}ms ({saving_pct:.0f}% of hot time)")
        if warm_avg > 0:
            ratio_cold_warm = cold_avg / warm_avg
            print(f"Cold is {ratio_cold_warm:.1f}x slower than Warm")
    
    return results


async def main():
    models = [
        ("Llama-3.2-1B", MODEL_1B),
        ("SmolLM2-1.7B", MODEL_1_7B),
    ]
    
    port_base = 19000
    
    for model_name, model_path in models:
        await run_benchmark(model_name, model_path, port=port_base)
        port_base += 1


if __name__ == "__main__":
    asyncio.run(main())
