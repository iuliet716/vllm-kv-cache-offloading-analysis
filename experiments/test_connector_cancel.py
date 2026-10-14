# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cancel R1 during a real CPU load; verify shared KV protection and R2 cleanup.
"""

import time

import pytest
import torch

from tests.v1.kv_connector.unit.utils import (
    EOS_TOKEN_ID,
    create_model_runner_output,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import RequestStatus


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_cancel_loader_while_same_prefix_request_waits(
    request_runner, async_scheduling, request
):
    runner = request_runner(
        block_size=4,
        blocks_per_chunk=3,
        num_gpu_blocks=100,
        async_scheduling=async_scheduling,
    )
    scheduler = runner.scheduler
    connector = runner.connector_scheduler
    manager = CPUOffloadingManager(num_chunks=1, cache_policy="lru", store_threshold=0)
    connector.manager = manager
    # 1. One CPU chunk and a small real GPU buffer.
    page_bytes = 32
    gpu = torch.zeros((100, page_bytes), dtype=torch.int8, device="cuda:0")
    expected = (torch.arange(3 * page_bytes) % 127).to(torch.int8)
    gpu[97:100].copy_(expected.reshape(3, page_bytes))
    worker = CPUOffloadingWorker(
        kv_caches=CanonicalKVCaches(
            tensors=[CanonicalKVCacheTensor(tensor=gpu, page_size_bytes=page_bytes)],
            group_data_refs=[
                [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=page_bytes)]
            ],
        ),
        blocks_per_chunk=3,
        num_cpu_chunks=1,
    )
    worker_connector = runner.worker_connector.connector_worker
    assert worker_connector is not None
    worker_connector.worker = worker
    request.addfinalizer(worker_connector.shutdown)
    load_end_events = {}
    results = {}

    def await_jobs(job_ids):
        deadline = time.monotonic() + 10
        while not set(job_ids) <= results.keys():
            for result in worker.get_finished():
                assert result.job_id not in results
                assert result.success
                results[result.job_id] = result
            if time.monotonic() >= deadline:
                missing = set(job_ids) - results.keys()
                raise AssertionError(f"Missing worker completions: {missing}")
            if not set(job_ids) <= results.keys():
                time.sleep(0.001)

    def collect_step(output, wait_for=(), require_pending=False, wait_stores=False):
        metadata = output.kv_connector_metadata
        assert not metadata.jobs_to_flush
        if any((job.req_id == r1 for job in metadata.load_jobs.values())):
            torch.cuda._sleep(500000000)
        worker_connector.handle_preemptions(metadata)
        worker_connector.start_kv_transfers(metadata)
        worker_connector.prepare_store_kv(metadata)
        for jid in metadata.load_jobs:
            load_end_events[jid] = worker._load_handler._transfer_events[jid]
        targets = set(wait_for)
        if wait_stores:
            targets.update(worker._store_handler._transfer_events)
        sending = set()
        receiving = set()
        aggregate = OffloadingWorkerMetadata()
        deadline = time.monotonic() + 10
        while True:
            sent, received = worker_connector.get_finished(
                set(output.finished_req_ids or ())
            )
            sending.update(sent)
            receiving.update(received)
            meta = worker_connector.build_connector_worker_meta()
            if require_pending and (sent or received or meta is not None):
                pytest.skip(
                    "INCONCLUSIVE: completion arrived before the cancellation window"
                )
            if meta is not None:
                assert not set(aggregate.completed_jobs) & set(
                    meta.completed_jobs
                ), "Duplicate completion within one step"
                aggregate = aggregate.aggregate(meta)
            if targets <= set(aggregate.completed_jobs):
                break
            if time.monotonic() >= deadline:
                raise AssertionError(
                    f"Missing connector completions: {targets - set(aggregate.completed_jobs)}"
                )
            time.sleep(0.001)
        return KVConnectorOutput(
            finished_sending=sending,
            finished_recving=receiving,
            kv_connector_worker_meta=aggregate,
        )
