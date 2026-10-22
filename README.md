# vLLM KV-Cache Offloading Analysis

Experiments on GPU–CPU KV-cache transfers and the request lifecycle in vLLM.

Tested with [vLLM](https://github.com/vllm-project/vllm) `0.30.1rc1.dev740+gaad4579bb` (development build),  
Python 3.12.15, pytest 9.1.1, PyTorch 2.13.0, NVIDIA RTX 5090, WSL2.

## Motivation

**KV-cache transfers run asynchronously and can outlive the request that started them.  
A missed completion can leave a request waiting; early cleanup can leave data unprotected.**

1. **Baseline:** Is evicted KV cache loaded back from CPU when a prompt is reused?
2. **Completion:** Is each transfer reported once per worker, and are stale reports ignored?
3. **Cancellation:** Does a canceled load keep its CPU data safe for a same-prefix request?

**Summary:** All 26 connector test cases passed without skips.  
Negative controls caught both injected faults: early chunk release and CPU-buffer overwrite.

## Experiments

### [GPU–CPU Transfers](experiments/check_transfer.py)

Seed a prompt, evict it from GPU with unrelated prompts, then reuse it.  
Compares transfer counters with expected KV-cache bytes, and compares outputs.

**Result:** 112 MiB was stored to CPU and loaded back on reuse, with the same output.

### [Completion Reporting](experiments/test_completion_delivery.py)

- `test_completion_is_reported_once`: once, for any completion order or read timing.
- `test_load_completes_only_after_all_workers_report`: only after every worker reports.
- `test_duplicate_completion_fails_loudly`: excess or repeated reports raise an error.
- `test_stale_completion_after_reset_is_ignored`: late reports after a reset are ignored.

**Result:** All 18 cases passed.  
Duplicates are caught only when the total exceeds the worker count,  
because the scheduler counts reports per job, not per worker.

### [Cancellation & Cleanup](experiments/test_connector_cancel.py)

Cancel R1 while its load is in flight and R2 waits on the same prefix.  
Runs with sync/async scheduling and pinned/pageable CPU memory.  
Async runs delay completion delivery by one step.

- `test_cancel_loader_while_same_prefix_request_waits`: chunk protection, GPU restore, cleanup.
- `test_detects_early_release_of_loading_chunk`: catches an early chunk release.
- `test_detects_overwrite_of_loading_chunk`: catches a CPU overwrite before the load reads it.

The overwrite control forces the Triton load path so CPU reads follow stream order.

**Result:** All 8 cases passed without skips.  
The 4 lifecycle cases held; the 4 negative controls caught their faults.

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

## Notes

- Cancellation extends upstream `test_abort_loading_requests` (mocked) with real transfers.
- Completion extends `test_complete_store_waits_for_all_worker_acks` and `test_reset_cache` to loads.
- Completion tests inject worker reports. Cancellation tests simulate model outputs.
- Async coverage is limited to completion timing.
- The baseline compares one generated token and does not measure performance.
