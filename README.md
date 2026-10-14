# vLLM KV-Cache Offloading Analysis

An investigation of vLLM CPU KV-cache offloading

# Scope

This repository documents experiments conducted in October 2026 using [vLLM](https://github.com/vllm-project/vllm) version `0.30.1rc1.dev740+gaad4579bb` (development build).

# Investigation

| Question | Method | Evidence |
| --- | --- | --- |
| Is KV cache really moved between GPU and CPU? | Check the server settings and transfer metrics | The server reported 112 MiB stored to CPU and 448 KiB loaded back |
| If one request is cancelled, can it free shared KV data too early? | Let R1 and R2 share the same prefix, cancel R1, and keep CPU cache space very limited | Test assertions confirm that the shared chunk remains referenced while R2 still needs it |
| Is a completed load reported to the scheduler only once? | Test different completion batches and metadata handling, including the real connector path | Mock and real connector tests verify that each completion is delivered once |
| After R1 is cancelled, can R2 still finish and clean up its state? | Continue R2, send EOS, and check that pending jobs and states are removed | Cleanup assertions confirm that R2 finishes and its pending state is released |

# Results

- Server logs recorded 112 MiB stored to CPU and 448 KiB loaded back.
- Completion-delivery tests passed all 6 cases.
- Cancellation tests passed with both scheduler variants: the in-flight CPU chunk remained protected, and R2 completed with request/job state cleaned up.

Test run: `8 passed, 14 warnings in 4.91s`.
