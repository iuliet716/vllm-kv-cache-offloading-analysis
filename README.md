# vLLM KV-Cache Offloading Analysis

Experiments on GPU–CPU KV-cache transfers and the request lifecycle in vLLM.

Tested with [vLLM](https://github.com/vllm-project/vllm) `0.30.1rc1.dev740+gaad4579bb` (development build).

## Experiments

### [GPU–CPU Transfers](experiments/check_transfer.py)
Seed a prompt, apply GPU cache pressure, and reuse the prompt to check whether its offloaded KV cache is loaded back from CPU. Checks transfer counters against expected KV-cache bytes and compares generated outputs.

### [Completion Reporting](experiments/test_completion_delivery.py)
Vary completion order and metadata read timing with a mocked transfer engine to check for missing or duplicate completion metadata.

**Result:** In all 6 cases, each job's completion metadata was consumed once, then cleared.

### [Cancellation & Cleanup](experiments/test_connector_cancel.py)
Cancel request R1 during a load while request R2 waits on the same prefix, and apply CPU cache pressure before and after cancellation. Runs with synchronous and asynchronous scheduling; skips if the load finishes before or during cancellation.

**Result:** In both modes, without skips, the in-flight CPU chunk remained protected from eviction and the restored GPU bytes matched the original data. R2 resumed and finished normally after an injected EOS, and all request and transfer state was cleared once pending stores completed.

## Run

Requires a CUDA environment with the tested vLLM source checkout and `pytest`.

```bash
# Connector tests
python run_test.py --vllm-dir /path/to/vllm --experiment all

# Server: 512 MiB GPU KV cache / 2 GiB CPU cache
bash run_server.sh > server.log 2>&1

# Once ready, run in another terminal
python experiments/check_transfer.py --output results/transfer-01
```
