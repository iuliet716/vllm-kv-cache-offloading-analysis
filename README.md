# vLLM KV-Cache Offloading Analysis

Experiments on GPU–CPU KV-cache transfers and the request lifecycle in vLLM.

Tested with [vLLM](https://github.com/vllm-project/vllm) `0.30.1rc1.dev740+gaad4579bb` (development build),  
Python 3.12.15, pytest 9.1.1, NVIDIA RTX 5090, PyTorch 2.13.0, WSL2.

## Motivation

**KV-cache offloading involves asynchronous transfers that can outlive a request.  
Missing completion reports or early cleanup could leave requests waiting or data unprotected.  
These experiments check completion reporting and cleanup.**

1. **Baseline:** When a prompt's KV cache is evicted from GPU memory, is it loaded back from CPU when the prompt is reused?
2. **Completion:** Is each transfer completion reported once per worker, and are stale reports ignored after a cache reset?
3. **Cancellation:** If a request is canceled while its load is in flight, is the CPU data kept safe for another request waiting on the same prefix?

**Summary:** All 26 connector test cases passed without skips. Negative controls detected both early chunk release and CPU-buffer overwrite.

## Experiments

### [GPU–CPU Transfers](experiments/check_transfer.py)
Seed a prompt, apply GPU cache pressure, and reuse the prompt to check whether its offloaded KV cache is loaded back from CPU.  
Checks transfer counters against expected KV-cache bytes and compares generated outputs.

**Result:** 112 MiB of KV cache was stored to CPU and loaded back on prefix reuse; the reused request produced the same output.

### [Completion Reporting](experiments/test_completion_delivery.py)

- `test_completion_is_reported_once`: Each completed transfer is reported once across different completion orders and metadata read timings.
- `test_load_completes_only_after_all_workers_report`: A load completes only after all workers report.
- `test_duplicate_completion_fails_loudly`: A duplicate report raises an error before the chunk is released again.
- `test_stale_completion_after_reset_is_ignored`: A late report for a job cleared by a cache reset is ignored.

**Result:** All 18 cases passed. Each worker must report each transfer exactly once; duplicate reports raise an error.

### [Cancellation & Cleanup](experiments/test_connector_cancel.py)

Cancel R1 while its load is pending and R2 waits on the same prefix. All tests run with sync and async scheduling.

- `test_cancel_loader_while_same_prefix_request_waits`: The CPU chunk stays protected, GPU data is restored, and R2 finishes with an injected EOS before cleanup is verified. Runs with pinned and pageable CPU memory.
- `test_detects_early_release_of_loading_chunk`: Detects an injected early chunk release.
- `test_detects_overwrite_of_loading_chunk`: Detects an injected CPU-buffer overwrite before the load reads it.

**Result:** All 8 cases passed without skips. The 4 lifecycle cases verified chunk protection, GPU data restoration, and cleanup. The 4 negative-control cases detected the injected faults.

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
