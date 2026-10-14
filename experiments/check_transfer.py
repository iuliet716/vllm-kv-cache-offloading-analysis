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
    found = set()
    # Match only byte counters, never histogram sums/buckets or time counters.
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
            found.add(match[1])
    return totals, found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for this run; must not already exist")
    parser.add_argument("--pressure-requests", type=int, default=8)
    parser.add_argument("--settle-seconds", type=float, default=15)
    args = parser.parse_args()
    if not 1 <= args.pressure_requests <= 100 or args.settle_seconds <= 0:
        parser.error("pressure-requests must be 1..100 and settle-seconds > 0")
    args.output.mkdir(parents=True, exist_ok=False)

    def call(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = Request(args.url.rstrip("/") + path, data=data,
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=300) as response:
            return response.read().decode()

    def snapshot(name):
        text = call("/metrics")
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

    call("/health")
    before, _ = snapshot("before")
    # Valid ordinary token IDs for the pinned Qwen tokenizer. Distinct first
    # blocks prevent the pressure requests from sharing the target's prefix.
    target = [100 + (i % 1000) for i in range(4096)]
    complete("seed", target)
    for index in range(args.pressure_requests):
        complete(f"pressure-{index:02d}", [2000 + index] * 4096)
    time.sleep(args.settle_seconds)
    snapshot("before-reuse")
    complete("reuse", target)
    deadline = time.monotonic() + args.settle_seconds
    while True:
        after, found = snapshot("after")
        delta = {key: after[key] - before[key] for key in before}
        if found == {"load", "store"} and all(value > 0 for value in delta.values()):
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(0.5)
    passed = found == {"load", "store"} and all(value > 0 for value in delta.values())
    result = {"before_bytes": before, "after_bytes": after,
              "delta_bytes": delta, "both_directions_observed": passed}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        print("INCONCLUSIVE: both directions were not observed. Inspect snapshots "
              "and server.log; check eviction, delayed metrics, and counter resets.",
              file=sys.stderr)
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
