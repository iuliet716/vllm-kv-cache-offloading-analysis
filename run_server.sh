#!/usr/bin/env bash

vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 --port 8000 --served-model-name kv-offload-test \
  --dtype bfloat16 --max-model-len 8192 --enforce-eager \
  --kv-cache-memory-bytes 536870912 --enable-prefix-caching \
  --max-num-batched-tokens 4096 --max-num-seqs 1 \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"CPUOffloadingSpec","cpu_bytes_to_use":2147483648,"eviction_policy":"lru","store_threshold":0}}'
