# vLLM KV-Cache Offloading Analysis

Experiments on GPU–CPU KV-cache transfers and the request lifecycle in vLLM.

Tested with [vLLM](https://github.com/vllm-project/vllm) `0.30.1rc1.dev740+gaad4579bb` (development build).

## Motivation

**KV-cache offloading involves asynchronous transfers that can outlive a request.  
Missing completion reports or early cleanup could leave requests waiting or data unprotected.  
These experiments check completion reporting and cleanup.**

1. **Baseline:** When a prompt's KV cache is evicted from GPU memory, is it loaded back from CPU when the prompt is reused?
2. **Completion:** Are completions reported without omissions or duplicates across different arrival orders and read timings?
3. **Cancellation:** If a request is canceled while loading KV cache, is the CPU data kept safe for another request waiting on the same prefix?

**Summary:** In the tested scenarios for (2) and (3), completions were reported once, in-flight data stayed protected, and cleanup completed.

## Experiments

### [GPU–CPU Transfers](experiments/check_transfer.py)
Seed a prompt, apply GPU cache pressure, and reuse the prompt to check whether its offloaded KV cache is loaded back from CPU. Checks transfer counters against expected KV-cache bytes and compares generated outputs.

**Result:** 112 MiB of KV cache was stored to CPU and loaded back on prefix reuse; the reused request produced the same output.

### [Completion Reporting](experiments/test_completion_delivery.py)
Vary completion order and metadata read timing with a mocked transfer engine to check for missing or duplicate completion metadata.

**Result:** In all 6 cases, each job's completion metadata was consumed once, then cleared.

### [Cancellation & Cleanup](experiments/test_connector_cancel.py)
Cancel request R1 during a load while request R2 waits on the same prefix, and apply CPU cache pressure before and after cancellation. Runs with synchronous and asynchronous scheduling; skips if the load finishes before or during cancellation.

**Result:** Both modes passed without skips, the in-flight CPU chunk remained protected from eviction and the restored GPU bytes matched the original data. R2 resumed and finished normally after an injected EOS, and the checked request and transfer states was cleared once pending stores completed.

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
