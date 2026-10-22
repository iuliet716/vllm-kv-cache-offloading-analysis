# vLLM KV-Cache Offloading Analysis

Experiments on GPU–CPU KV-cache transfers and the request lifecycle in vLLM.

Tested with [vLLM](https://github.com/vllm-project/vllm) `0.30.1rc1.dev740+gaad4579bb` (development build),  
Python 3.12.15, pytest 9.1.1, PyTorch 2.13.0, NVIDIA RTX 5090, WSL2.

## Motivation

**KV-cache transfers run asynchronously and can outlive the request that started them.  
A missed completion can leave a request waiting; early cleanup can leave data unprotected.**

1. **Baseline:** Is evicted KV cache loaded back from CPU when a prompt is reused?
2. **Completion:** Is each transfer reported once per worker, and are stale reports ignored after a cache reset?
3. **Cancellation:** If a request is canceled mid-load, is the CPU data safe for a request waiting on the same prefix?

**Summary:** All 26 connector test cases passed without skips.  
Negative controls caught both injected faults: early chunk release and CPU-buffer overwrite.

## Experiments

### [GPU–CPU Transfers](experiments/check_transfer.py)

Serve Qwen2.5-1.5B-Instruct with a 512 MiB GPU KV cache and a 2 GiB CPU cache.  
Seed a 4,096-token prompt, evict it from GPU with unrelated prompts, then reuse it.  
Passes if transfer counters reach 90% of the expected KV-cache bytes and the outputs match.

**Result:** 112 MiB of KV cache was stored to CPU and loaded back on reuse; the output matched.

### [Completion Reporting](experiments/test_completion_delivery.py)

Worker-side tests use a mocked transfer engine.  
Scheduler-side tests feed per-worker completion counts directly to the connector.

- `test_completion_is_reported_once`: reported once across 3 arrival orders and 2 read timings.
- `test_load_completes_only_after_all_workers_report`: completes only after every worker reports.
- `test_duplicate_completion_fails_loudly`: excess reports, or reports for a finished job, raise an error.
- `test_stale_completion_after_reset_is_ignored`: a late report for a job dropped by a reset is ignored.

**Result:** All 18 cases passed.  
Duplicates are caught only when the total exceeds the worker count, since the scheduler counts per job.  
Each worker must therefore report each transfer exactly once.

### [Cancellation & Cleanup](experiments/test_connector_cancel.py)

Cancel R1 while its load is in flight and R2 waits on the same prefix.  
A CUDA stream stall keeps the load in flight; the test skips if it finishes before cancellation.  
Uses a real CPU offloading manager and transfer worker, with simulated model outputs.  
Runs with sync and async scheduling; async runs delay completion delivery by one step.

- `test_cancel_loader_while_same_prefix_request_waits` (pinned and pageable CPU memory):
  - Eviction pressure cannot reclaim the CPU chunk before or after cancellation.
  - Restored GPU bytes match the seeded data.
  - With async scheduling, the chunk stays protected until the delayed completion arrives.
  - R2 resumes and finishes; all request and transfer state drains.
- `test_detects_early_release_of_loading_chunk`: catches an injected early chunk release.
- `test_detects_overwrite_of_loading_chunk`: catches an injected CPU overwrite before the load reads it.

The overwrite control forces the Triton load path so CPU reads follow stream order.

**Result:** All 8 cases passed without skips: 4 lifecycle cases and 4 negative controls.

## Run

Requires a CUDA environment with the tested vLLM source checkout and `pytest`.  
`run_test.py` runs the tests inside vLLM's test tree to reuse its upstream fixtures.

```bash
# Connector tests
python run_test.py --vllm-dir /path/to/vllm --experiment all

# Server: 512 MiB GPU KV cache / 2 GiB CPU cache
bash run_server.sh > server.log 2>&1

# Once ready, run in another terminal
python experiments/check_transfer.py --output results/transfer-01
```

## Notes

- Cancellation extends upstream `test_abort_loading_requests`, which mocks the manager and transfers.  
  It adds real transfers, a same-prefix waiter, eviction pressure, byte checks, and negative controls.
- Completion extends `test_complete_store_waits_for_all_worker_acks` and `test_reset_cache`  
  to load jobs and to late reports after a reset.
- Async coverage is limited to completion timing; model outputs are applied immediately.
- The baseline compares one generated token and does not measure performance.
