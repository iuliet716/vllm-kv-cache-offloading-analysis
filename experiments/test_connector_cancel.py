# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
from vllm.v1.kv_offload.cpu.swap_blocks_triton import (
    swap_blocks_batch as triton_swap_blocks_batch,
)
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import RequestStatus


@pytest.mark.parametrize("pin_memory", [False, True])
@pytest.mark.parametrize("async_scheduling", [False, True])
def test_cancel_loader_while_same_prefix_request_waits(
    request_runner, async_scheduling, pin_memory, request, monkeypatch
):
    _run_cancel_scenario(
        request_runner, async_scheduling, request, monkeypatch, pin_memory
    )


def _run_cancel_scenario(
    request_runner, async_scheduling, request, monkeypatch, pin_memory, mutation=None
):
    # At this size (3 descriptors), loads use the C++ batch copy in both memory modes; only the overwrite control forces the Triton kernel.
    # Both memory modes are tested for chunk protection and lifecycle.
    monkeypatch.setattr("vllm.v1.kv_offload.cpu.gpu_worker.PIN_MEMORY", pin_memory)
    runner = request_runner(
        block_size=4,
        blocks_per_chunk=3,
        num_gpu_blocks=100,
        async_scheduling=async_scheduling,
    )
    scheduler = runner.scheduler
    connector = runner.connector_scheduler

    # Setup: one CPU chunk, a real GPU tensor, and a real transfer worker
    manager = CPUOffloadingManager(num_chunks=1, cache_policy="lru", store_threshold=0)
    connector.manager = manager
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
    if worker._load_handler.src_tensors[0].is_pinned() != pin_memory:
        pytest.skip(f"Could not allocate CPU buffer with pin_memory={pin_memory}")
    load_end_events = {}
    results = {}

    # Wait until the copy worker finishes the given jobs (seed store, warm-up load)
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

    # Simulate one worker step: start transfers and collect completions
    def collect_step(output, wait_for=(), require_pending=False, wait_stores=False):
        metadata = output.kv_connector_metadata
        assert not metadata.jobs_to_flush
        # Stall the GPU so R1's load is still in flight when R1 is canceled
        if any(job.req_id == r1 for job in metadata.load_jobs.values()):
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
            # Skip if the load finished too early
            if require_pending and (sent or received or meta is not None):
                pytest.skip("Completion arrived before the cancellation window")
            if meta is not None:
                assert not set(aggregate.completed_jobs) & set(meta.completed_jobs), (
                    "Duplicate completion within one step"
                )
                aggregate = aggregate.aggregate(meta)
            if targets <= set(aggregate.completed_jobs):
                break
            if time.monotonic() >= deadline:
                missing = targets - set(aggregate.completed_jobs)
                raise AssertionError(f"Missing connector completions: {missing}")
            time.sleep(0.001)
        return KVConnectorOutput(
            finished_sending=sending,
            finished_recving=receiving,
            kv_connector_worker_meta=aggregate,
        )

    # Async scheduling applies step N's output only after step N+1 is scheduled. 
    # Model this by holding outputs until the next schedule().
    held = []

    def flush():
        while held:
            scheduler.update_from_output(*held.pop(0))

    def schedule():
        output = scheduler.schedule()
        flush()
        return output

    # Run one worker step and return its output to the scheduler (held if async)
    def report(output, wait_for=(), require_pending=False, wait_stores=False):
        actual = collect_step(
            output,
            wait_for=wait_for,
            require_pending=require_pending,
            wait_stores=wait_stores,
        )
        model_output = ModelRunnerOutput.with_kv_conn_output_only(actual)
        if async_scheduling:
            held.append((output, model_output))
        else:
            scheduler.update_from_output(output, model_output)

    # 1. Seed the CPU with a known pattern, then clear the GPU
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
    manager.complete_store(stored.keys_to_store, seed)
    manager.on_request_finished(seed)
    # Warm up the load path so kernel compilation cannot outlast the GPU stall
    warm = ReqContext(req_id="warmup")
    manager.on_new_request(warm)
    src = manager.prepare_load(keys, warm)
    assert worker.submit_load(
        -2, src, GPULoadStoreSpec([0, 1, 2], group_sizes=(3,), block_indices=(0,))
    )
    await_jobs({-2})
    assert torch.equal(gpu[0:3].cpu().reshape(-1), expected)
    manager.complete_load(keys, warm)
    manager.on_request_finished(warm)
    gpu.zero_()
    torch.cuda.synchronize()

    # 2. Schedule R1: one load job, chunk referenced once
    first = schedule()
    assert not first.num_scheduled_tokens
    loads = first.kv_connector_metadata.load_jobs
    assert len(loads) == 1
    job_id = next(iter(loads))
    assert connector._jobs[job_id].req_id == r1
    chunk = manager._policy.get(keys[0])
    assert chunk is not None and chunk.ref_cnt == 1
    report(first, require_pending=True)

    # 3. Add R2 with the same prompt: it waits without a duplicate load
    runner.new_request(token_ids=prompt)
    r2 = str(runner.req_id)
    waiting = schedule()
    assert not waiting.num_scheduled_tokens
    assert not waiting.kv_connector_metadata.load_jobs
    assert r2 in scheduler.requests
    assert keys[0] in connector._chunks_being_loaded
    assert chunk.ref_cnt == 1
    report(waiting, require_pending=True)

    # Storing a different key must fail while the chunk is in use
    pressure_key = make_offload_key(b"pressure-only", 0)

    def check_pressure_blocked(label):
        ctx = ReqContext(req_id=label)
        manager.on_new_request(ctx)
        assert manager.prepare_store([pressure_key], ctx) is None, (
            f"{label}: pressure store evicted the in-flight chunk"
        )
        manager.on_request_finished(ctx)
        assert manager._policy.get(keys[0]) is chunk
        assert chunk.ref_cnt == 1
        assert manager._policy.get(pressure_key) is None

    # 4. Pressure before cancellation: the chunk must stay
    check_pressure_blocked("before-cancel")

    # 5. Cancel R1 while its load is still running
    end_event = load_end_events[job_id]
    if end_event.query():
        pytest.skip("R1 load finished before cancellation")
    scheduler.finish_requests((r1,), RequestStatus.FINISHED_ABORTED)
    if end_event.query():
        pytest.skip("R1 load finished during cancellation")
    # Fault injection for negative-control tests
    if mutation is not None:
        mutation(manager=manager, worker=worker, state=state, keys=keys, job_id=job_id)
    # Behavioral check first: pressure must not evict the in-flight chunk
    check_pressure_blocked("after-cancel")
    # R1 removal is deferred and the chunk stays protected
    assert r1 in scheduler.requests
    assert r2 in scheduler.requests
    assert not scheduler.requests[r2].is_finished()
    assert chunk.ref_cnt == 1
    assert job_id in connector._jobs

    # 6. Finish the load: bytes restored, then R1 is released
    pending = schedule()
    assert not pending.num_scheduled_tokens
    assert not pending.kv_connector_metadata.load_jobs
    report(pending, wait_for={job_id})
    assert end_event.query()
    loaded_blocks = first.kv_connector_metadata.load_jobs[job_id].dst_spec.block_ids
    for offset, block_id in enumerate(loaded_blocks):
        assert torch.equal(
            gpu[int(block_id)].cpu(), expected.reshape(3, page_bytes)[offset]
        ), f"Restored GPU block {int(block_id)} differs from the seeded data"
    assert torch.equal(worker._load_handler.src_tensors[0], cpu_before)

    if async_scheduling:
        # 6a. The load is done on the worker, but the scheduler has not received the completion yet.
        # The chunk must stay protected and R2 must wait
        assert job_id in connector._jobs
        assert chunk.ref_cnt == 1
        assert keys[0] in connector._chunks_being_loaded
        assert r1 in scheduler.requests
        check_pressure_blocked("completion-in-flight")
        lagged = schedule()  # schedules first, then applies the completion
        assert not lagged.num_scheduled_tokens, (
            "R2 ran before the load completion reached the scheduler"
        )
        assert not lagged.kv_connector_metadata.load_jobs
        report(lagged)

    assert job_id not in connector._jobs
    assert chunk.ref_cnt == 0
    assert keys[0] not in connector._chunks_being_loaded
    assert r1 not in scheduler.requests

    # 7. Resume R2 and finish it with an injected EOS
    for _ in range(6):
        output = schedule()
        loads = output.kv_connector_metadata.load_jobs
        if output.num_scheduled_tokens.get(r2, 0) > 0:
            assert not loads
            assert r1 not in connector._req_status
            assert chunk.ref_cnt == 0
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
            # EOS is applied immediately in both modes; only the transfer completion lag is modeled
            scheduler.update_from_output(output, model_output)
            assert survivor.is_finished()
            assert survivor.status != RequestStatus.FINISHED_ABORTED

            # 8. Drain pending stores; all state must be cleared
            for _ in range(8):
                drained = schedule()
                assert not drained.num_scheduled_tokens
                assert not drained.kv_connector_metadata.load_jobs
                for job in connector._jobs.values():
                    assert job.is_store, "Unexpected unfinished load"
                report(drained, wait_stores=True)
                if not (
                    scheduler.requests
                    or scheduler.finished_req_ids
                    or connector._jobs
                    or connector._req_status
                ):
                    break
            else:
                raise AssertionError("Request/job state did not drain")
            flush()
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
            return
        # R2 may load its own blocks before it runs
        assert not output.num_scheduled_tokens
        for jid in loads:
            assert connector._jobs[jid].req_id == r2
        report(output, wait_for=set(loads))
    raise AssertionError("R2 did not resume within six controlled steps")


# Negative controls: each injects one fault and expects a specific check to fail


def _release_early(*, manager, state, keys, **_):
    # Bug: release R1's hold on the chunk at abort instead of when the load completes
    manager.complete_load(keys, state.req_context)


def _overwrite_cpu_chunk(*, worker, job_id, **_):
    # Simulate the CPU chunk being overwritten before the load reads it.
    # Require the Triton path so the stream wait gates CPU-memory reads.
    handler = worker._load_handler
    if getattr(handler._swap_blocks_batch, "func", None) is not triton_swap_blocks_batch:
        pytest.skip("Load does not use the Triton path; CPU read timing is unordered")
    transfer = next(t for t in handler._transfers if t.job_id == job_id)
    if transfer.start_event.query():
        pytest.skip("R1 load started before the overwrite")
    handler.src_tensors[0].fill_(-1)
    if transfer.start_event.query():
        pytest.skip("R1 load may have started during the overwrite")


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_detects_early_release_of_loading_chunk(
    request_runner, async_scheduling, request, monkeypatch
):
    # Bookkeeping only, so the memory mode does not matter
    with pytest.raises(AssertionError, match="evicted the in-flight chunk"):
        _run_cancel_scenario(
            request_runner,
            async_scheduling,
            request,
            monkeypatch,
            pin_memory=False,
            mutation=_release_early,
        )


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_detects_overwrite_of_loading_chunk(
    request_runner, async_scheduling, request, monkeypatch
):
    # Force the Triton kernel for this three-descriptor negative control.
    # Otherwise the C++ batch copy runs with SRC_ACCESS_ORDER_ANY and may read CPU data before the stream reaches start_event.
    monkeypatch.setattr("vllm.v1.kv_offload.cpu.swap_blocks_triton.MIN_N", 1)
    with pytest.raises(AssertionError, match="differs from the seeded data"):
        _run_cancel_scenario(
            request_runner,
            async_scheduling,
            request,
            monkeypatch,
            pin_memory=True,
            mutation=_overwrite_cpu_chunk,
        )
