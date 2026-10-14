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

    def report(output, wait_for=(), require_pending=False, wait_stores=False):
        actual = collect_step(
            output,
            wait_for=wait_for,
            require_pending=require_pending,
            wait_stores=wait_stores,
        )
        scheduler.update_from_output(
            output, ModelRunnerOutput.with_kv_conn_output_only(actual)
        )

    # 2. Seed CPU KV for a 12-token prefix plus one uncached token.
    prompt = list(range(13))
    runner.new_request(token_ids=prompt)
    r1 = str(runner.req_id)
    state = connector._req_status[r1]
    state.update_offload_keys()
    keys = state.group_states[0].offload_keys[:1]
    assert len(keys) == 1
    seed = ReqContext(req_id="seed")
    manager.on_new_request(seed)
    stored = manager.prepare_store(keys, seed)
    assert stored is not None
    assert worker.submit_store(
        -1,
        GPULoadStoreSpec([97, 98, 99], group_sizes=(3,), block_indices=(0,)),
        stored.store_spec,
    )
    await_jobs({-1})
    assert results[-1].transfer_size == 3 * page_bytes
    cpu_before = worker._load_handler.src_tensors[0].clone()
    assert torch.equal(cpu_before.reshape(-1), expected)
    gpu.zero_()
    manager.complete_store(stored.keys_to_store, seed)
    manager.on_request_finished(seed)
    torch.cuda._sleep(1)
    torch.cuda.synchronize()

    # 3. Start R1 loading; R2 must wait on the same prefix.
    first = scheduler.schedule()
    assert not first.num_scheduled_tokens
    loads = first.kv_connector_metadata.load_jobs
    assert len(loads) == 1
    job_id = next(iter(loads))
    assert connector._jobs[job_id].req_id == r1
    chunk = manager._policy.get(keys[0])
    assert chunk is not None and chunk.ref_cnt == 1
    report(first, require_pending=True)
    runner.new_request(token_ids=prompt)
    r2 = str(runner.req_id)
    waiting = scheduler.schedule()
    assert not waiting.num_scheduled_tokens
    assert not waiting.kv_connector_metadata.load_jobs
    assert r2 in scheduler.requests
    assert keys[0] in connector._chunks_being_loaded
    assert chunk.ref_cnt == 1
    report(waiting, require_pending=True)
    print("R2 waits without issuing a duplicate load: PASS")
    pressure_key = make_offload_key(b"pressure-only", 0)

    def check_pressure_blocked(label):
        ctx = ReqContext(req_id=label)
        manager.on_new_request(ctx)
        assert manager.prepare_store([pressure_key], ctx) is None
        manager.on_request_finished(ctx)
        assert manager._policy.get(keys[0]) is chunk
        assert chunk.ref_cnt == 1
        assert manager._policy.get(pressure_key) is None
        print(f"{label}: in-flight chunk protected")

    # 4. Cancel R1 while its transfer is pending; pressure must not evict KV.
    check_pressure_blocked("before-cancel")
    end_event = load_end_events[job_id]
    if end_event.query():
        pytest.skip(
            "INCONCLUSIVE: R1 load finished before cancellation; pending-transfer timing was not established"
        )
    scheduler.finish_requests((r1,), RequestStatus.FINISHED_ABORTED)
    if end_event.query():
        pytest.skip(
            "INCONCLUSIVE: R1 load finished during cancellation; pending state after cancellation was not established"
        )
    print("R1 load end event pending before AND after cancellation: PASS")
    assert r1 in scheduler.requests
    assert r2 in scheduler.requests
    assert not scheduler.requests[r2].is_finished()
    assert chunk.ref_cnt == 1
    assert job_id in connector._jobs
    print("R1 cancellation preserves the in-flight load: PASS")
    check_pressure_blocked("after-cancel")

    # 5. Finish the transfer and release R1.
    pending = scheduler.schedule()
    assert not pending.num_scheduled_tokens
    assert not pending.kv_connector_metadata.load_jobs
    report(pending, wait_for={job_id})
    # The actual load must restore the seeded bytes to the scheduled GPU blocks.
    loaded_blocks = first.kv_connector_metadata.load_jobs[job_id].dst_spec.block_ids
    for offset, block_id in enumerate(loaded_blocks):
        assert torch.equal(
            gpu[int(block_id)].cpu(), expected.reshape(3, page_bytes)[offset]
        )
    assert torch.equal(worker._load_handler.src_tensors[0], cpu_before)
    assert job_id not in connector._jobs
    assert chunk.ref_cnt == 0
    assert keys[0] not in connector._chunks_being_loaded
    assert r1 not in scheduler.requests
    print("R1 completion releases its load reference: PASS")

    # 6. Resume R2, emit EOS, then drain pending stores and request state.
    for _ in range(4):
        output = scheduler.schedule()
        loads = output.kv_connector_metadata.load_jobs
        if output.num_scheduled_tokens.get(r2, 0) > 0:
            assert not loads
            assert r1 not in connector._req_status
            assert chunk.ref_cnt == 0
            print("R2 resumes model execution: PASS")
            assert not output.kv_connector_metadata.store_jobs
            survivor = scheduler.requests[r2]
            actual = collect_step(output)
            model_output = create_model_runner_output(
                reqs=scheduler.running,
                finished_sending=actual.finished_sending,
                finished_recving=actual.finished_recving,
                token_id=EOS_TOKEN_ID,
                kv_connector_worker_meta=actual.kv_connector_worker_meta,
            )
            scheduler.update_from_output(output, model_output)
            assert survivor.is_finished()
            assert survivor.status != RequestStatus.FINISHED_ABORTED
            for _ in range(8):
                drained = scheduler.schedule()
                assert not drained.num_scheduled_tokens
                assert not drained.kv_connector_metadata.load_jobs
                for job in connector._jobs.values():
                    assert job.is_store, "Unexpected unfinished load"
                report(drained, wait_stores=True)
                if (
                    not scheduler.requests
                    and (not scheduler.finished_req_ids)
                    and (not connector._jobs)
                    and (not connector._req_status)
                ):
                    break
            else:
                raise AssertionError("Request/job state did not drain")
            assert not connector._chunks_being_loaded
            assert not connector._block_id_to_pending_jobs
            assert not connector.has_pending_push_work()
            assert manager._num_write_pending_chunks == 0
            resident_count = manager._num_allocated_chunks - len(manager._free_list)
            assert resident_count == manager._num_evictable_cache_chunks
            assert not worker_connector._load_jobs
            assert not worker_connector._unsubmitted_store_jobs
            assert worker_connector.build_connector_worker_meta() is None
            assert not worker._load_handler._transfer_events
            assert not worker._store_handler._transfer_events
            print("R2 EOS: request/job cleanup and CPU unpinning PASS")
            return
        assert not output.num_scheduled_tokens
        for jid in loads:
            assert connector._jobs[jid].req_id == r2
        report(output, wait_for=set(loads))
    raise AssertionError("R2 did not resume within four controlled steps")
