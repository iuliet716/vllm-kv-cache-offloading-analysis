# vLLM KV-Cache Offloading Analysis

Experiments on GPU–CPU KV-cache transfers, completion delivery, and request cancellation in vLLM.

## Scope

Tested in October 2026 with [vLLM](https://github.com/vllm-project/vllm) version `0.30.1rc1.dev740+gaad4579bb` (development build).

These experiments verify specific offloading lifecycle scenarios, not serving performance or correctness across all workloads.

## Investigation

| Question | Method | Result |
| --- | --- | --- |
| Does KV cache move between GPU and CPU? | Run a server with CPU offloading and inspect transfer metrics | An earlier server run reported 112 MiB stored to CPU and 448 KiB loaded back |
| Is load completion metadata consumed only once? | Test three completion batch patterns with two metadata-reading strategies | All 6 cases passed; completion metadata was consumed once and cleared |
| Can cancelling R1 release an in-flight CPU chunk too early? | Queue R2 with the same prefix, cancel R1 during a real load, and apply storage pressure with one CPU chunk available | The chunk remained referenced and protected from eviction until the transfer completed |
| Can R2 resume and clean up its state? | Complete the transfer, verify restored GPU bytes, resume R2, inject EOS, and drain pending work | R2 finished and request/job state, pending references, and transfer events were cleared |

## Results

All **8 test cases passed without skips**:

- 6 completion-delivery cases.
- 2 cancellation cases covering synchronous and asynchronous scheduling.

Both cancellation cases confirmed that the load was still pending immediately before and after R1 cancellation.

The server transfer figures above come from an earlier run, separate from these 8 tests.

## Test coverage

- [`test_completion_delivery.py`](experiments/test_completion_delivery.py): real connector logic with mocked worker completion results.
- [`test_connector_cancel.py`](experiments/test_connector_cancel.py): scheduler–connector–worker integration with real GPU–CPU transfers and byte comparisons. Model output and EOS are supplied by the test.
- [`check_transfer.py`](experiments/check_transfer.py): server requests and transfer counter checks, with request/response JSON, metric snapshots, and a summary saved for inspection.

The cancellation test skips if the transfer finishes before the required cancellation window can be established.

## Run

Use a CUDA-capable environment with the tested vLLM source checkout, its runtime and test dependencies, and `pytest` installed.

The tests depend on upstream fixtures and private helpers. Other vLLM versions may require changes.

Run the following commands from this repository using the Python environment prepared for vLLM.

### Connector tests

```bash
python run_test.py --vllm-dir /path/to/vllm --experiment all
```

Use `--experiment completion` or `--experiment cancel` to run one group.

### Server transfer check

Start the Qwen2.5-1.5B-Instruct server with a 512 MiB GPU KV cache and a 2 GiB CPU cache:

```bash
bash run_server.sh > server.log 2>&1
```

Once the server is ready, run in another terminal:

```bash
python experiments/check_transfer.py --output results/transfer-01
```

The output directory must not already exist.

Inspect `summary.json` and the saved metric snapshots. Exit code `0` indicates positive load and store counter deltas; `2` indicates an inconclusive result.
