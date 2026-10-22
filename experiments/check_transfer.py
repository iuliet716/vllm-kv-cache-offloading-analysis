import argparse
import json
import math
from pathlib import Path
import re
import sys
import time
from urllib.request import Request, urlopen


def byte_counters(text):
    totals = {"load": 0.0, "store": 0.0}
    pattern = re.compile(
        r'^vllm:kv_offload_(load|store)_bytes(?:_total)?(?:\{.*\})?\s+(\S+)'
    )
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            value = float(match[2])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid counter: {line}")
            totals[match[1]] += value
    return totals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for this run; must not already exist")
    parser.add_argument("--pressure-requests", type=int, default=8)
    parser.add_argument("--settle-seconds", type=float, default=15)
    args = parser.parse_args()
    if not 1 <= args.pressure_requests <= 100 or args.settle_seconds <= 0:
        parser.error("pressure-requests must be 1..100 and settle-seconds > 0")
    args.output.mkdir(parents=True, exist_ok=False)

    def call(path, payload=None, timeout=300):
        data = None if payload is None else json.dumps(payload).encode()
        request = Request(args.url.rstrip("/") + path, data=data,
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode()

    def snapshot(name):
        text = call("/metrics", timeout=5)
        (args.output / f"{name}.prom").write_text(text)
        return byte_counters(text)

    def complete(name, prompt):
        payload = {"model": "kv-offload-test", "prompt": prompt,
                   "max_tokens": 1, "temperature": 0}
        (args.output / f"{name}.request.json").write_text(json.dumps(payload))
        response = call("/v1/completions", payload)
        (args.output / f"{name}.response.json").write_text(response)
        parsed = json.loads(response)
        if not parsed.get("choices"):
            raise RuntimeError(f"No completion in {name}")
        return parsed

    def wait_delta(base, key, minimum, name):
        deadline = time.monotonic() + args.settle_seconds
        while True:
            now = snapshot(name)
            delta = now[key] - base[key]
            if delta >= minimum or time.monotonic() >= deadline:
                return delta
            time.sleep(0.5)

    # Qwen2.5-1.5B: 28 layers * K/V * 2 KV heads * 128 head_dim * 2 bytes (bf16)
    bytes_per_token = 28 * 2 * 2 * 128 * 2
    prompt_len, block_size, min_ratio = 4096, 16, 0.9
    expected_store = prompt_len * bytes_per_token
    expected_load = (prompt_len - block_size) * bytes_per_token

    # 0. Health Check 
    call("/health")
    before = snapshot("before")
    
    # 1. Request the target prompt; its KV-cache is stored to CPU
    target = [100 + (i % 1000) for i in range(prompt_len)]
    seed = complete("seed", target)
    store_delta = wait_delta(before, "store", min_ratio * expected_store, "after-seed")
    
    # 2. Request unrelated prompts to fill the GPU cache and evict the target
    for index in range(args.pressure_requests):
        complete(f"pressure-{index:02d}", [2000 + index] * prompt_len)
    time.sleep(args.settle_seconds)
    before_reuse = snapshot("before-reuse")
    
    # 3. Re-request the target prompt to load its KV-cache back from CPU
    reuse = complete("reuse", target)
    load_delta = wait_delta(before_reuse, "load", min_ratio * expected_load, "after")

    # 4. Judge each step against expected bytes and compare outputs.
    checks = {
        "seed_stored": store_delta >= min_ratio * expected_store,
        "reuse_loaded": load_delta >= min_ratio * expected_load,
        "same_output": seed["choices"][0]["text"] == reuse["choices"][0]["text"],
    }
    result = {"expected_store_bytes": expected_store, "store_delta_bytes": store_delta,
              "expected_load_bytes": expected_load, "load_delta_bytes": load_delta,
              "checks": checks, "passed": all(checks.values())}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
