# vLLM KV-Cache Offloading Analysis

Experiments on GPU–CPU KV-cache transfers and the request lifecycle in vLLM.

Tested with [vLLM](https://github.com/vllm-project/vllm) `0.30.1rc1.dev740+gaad4579bb` (development build).

## Experiments

### [GPU–CPU Transfers](experiments/check_transfer.py)
Seed a prompt, apply GPU cache pressure, and reuse the prompt to check whether offloaded KV-cache is loaded back from CPU.


### [Completion Reporting](experiments/test_completion_delivery.py)
Vary completion order and metadata read timing to check for missing or duplicate completion reports.

**Result:** Each job’s completion metadata was consumed once, then cleared.

### [Cancellation & Cleanup](experiments/test_connector_cancel.py)
Cancel request R1 during a load while request R2 waits on the same prefix, then apply CPU cache pressure.  
Runs with synchronous and asynchronous scheduling; skips if the load finishes before or during cancellation.

**Result:** The in-flight CPU chunk remained protected from eviction. Restored GPU bytes matched the original data, R2 resumed, and request/transfer state was cleared after injected EOS and pending work completed.

## Run

```bash
# Connector tests
python run_test.py --vllm-dir /path/to/vllm --experiment all

# Server: 512 MiB GPU KV cache / 2 GiB CPU cache
bash run_server.sh > server.log 2>&1

# Once ready, run in another terminal
python experiments/check_transfer.py --output results/transfer-01
```
