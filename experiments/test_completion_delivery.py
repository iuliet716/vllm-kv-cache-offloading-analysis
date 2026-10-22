import pytest

from tests.v1.kv_connector.unit.offloading_connector.test_worker import _make_worker
from tests.v1.kv_connector.unit.offloading_connector.utils import (
    generate_store_output,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    TransferJob,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import GPULoadStoreSpec, LoadStoreSpec, TransferResult
from vllm.v1.outputs import KVConnectorOutput

# ---------------------------------------------------------------------------
# Worker side: each completion leaves the worker exactly once
# ---------------------------------------------------------------------------

# One load job per request; the scheduler never issues two at once
JOB_REQS = {10: "req-a", 11: "req-b"}


def _load_job_metadata(job_id, req_id):
    return OffloadingConnectorMetadata(
        load_jobs={
            job_id: TransferJob(
                req_id=req_id,
                src_spec=LoadStoreSpec(),
                dst_spec=GPULoadStoreSpec([0], group_sizes=(1,), block_indices=(0,)),
            )
        },
        store_jobs={},
    )


# Completion arrival: all at once / one per step / empty steps in reverse order
@pytest.mark.parametrize("batches", [[[10, 11]], [[10], [11]], [[], [11], [], [10]]])
# Metadata reading: once at the end / after every step
@pytest.mark.parametrize("read_each_batch", [False, True])
def test_completion_is_reported_once(batches, read_each_batch):
    # 1. Create a worker connector with a mocked transfer engine (no GPU)
    connector, _ = _make_worker(
        KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[])
    )
    # 2. Start one load job for each of two requests
    for job_id, req_id in JOB_REQS.items():
        connector.start_kv_transfers(_load_job_metadata(job_id, req_id))

    delivered = []
    for batch in batches:
        # 3. Inject this step's completions and collect them
        connector.worker.get_finished.return_value = [
            TransferResult(job_id, True) for job_id in batch
        ]
        sent, received = connector.get_finished(set())
        assert sent == set()
        assert received == {JOB_REQS[job_id] for job_id in batch}

        # 4a. Read every step: only this step's jobs, then the buffer is empty
        if read_each_batch:
            metadata = connector.build_connector_worker_meta()
            if metadata is not None:
                assert metadata.completed_jobs == {job_id: 1 for job_id in batch}
                delivered.extend(metadata.completed_jobs)
            else:
                assert not batch
            assert connector.build_connector_worker_meta() is None

    # 4b. Read once at the end: all accumulated jobs, each exactly once
    if not read_each_batch:
        metadata = connector.build_connector_worker_meta()
        assert metadata.completed_jobs == {10: 1, 11: 1}
        delivered.extend(metadata.completed_jobs)

    # 5. Each job is delivered once; load tracking and completion metadata are empty
    assert sorted(delivered) == [10, 11]
    assert not connector._load_jobs
    connector.worker.get_finished.return_value = []
    assert connector.get_finished(set()) == (set(), set())
    assert connector.build_connector_worker_meta() is None


# ---------------------------------------------------------------------------
# Scheduler side: per-worker completion counts, including duplicates
# ---------------------------------------------------------------------------


def _worker_meta(completed_jobs):
    return KVConnectorOutput(
        kv_connector_worker_meta=OffloadingWorkerMetadata(completed_jobs=completed_jobs)
    )


def _start_inflight_load(request_runner, async_scheduling, worker_count):
    runner = request_runner(
        block_size=4,
        num_gpu_blocks=100,
        async_scheduling=async_scheduling,
        blocks_per_chunk=3,
        worker_count=worker_count,
    )
    # Force a one-chunk CPU hit for a 12-token request and leave its load unfinished
    runner.new_request(token_ids=[0] * 12)
    runner.connector_scheduler._maximal_prefix_lookup = lambda keys, ctx, *_: 1
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.run(decoded_tokens=[], complete_transfers=False)

    connector = runner.connector_scheduler
    loads = [jid for jid, job in connector._jobs.items() if not job.is_store]
    assert len(loads) == 1
    job_id = loads[0]
    keys = set(connector._jobs[job_id].keys)
    assert connector._chunks_being_loaded is not None
    assert keys and keys <= connector._chunks_being_loaded
    assert connector._jobs[job_id].pending_count == worker_count
    runner.manager.reset_mock()
    return runner, job_id, keys


# (worker_count, per-delivery counts): all at once, or split across steps
@pytest.mark.parametrize(
    "worker_count, deliveries",
    [(2, [2]), (2, [1, 1]), (3, [1, 2]), (3, [1, 1, 1])],
)
@pytest.mark.parametrize("async_scheduling", [False, True])
def test_load_completes_only_after_all_workers_report(
    request_runner, async_scheduling, worker_count, deliveries
):
    runner, job_id, keys = _start_inflight_load(
        request_runner, async_scheduling, worker_count
    )
    connector = runner.connector_scheduler

    remaining = worker_count
    for count in deliveries:
        connector.update_connector_output(_worker_meta({job_id: count}))
        remaining -= count
        if remaining:
            # Partial count: keep the job open and its chunks marked as loading
            assert connector._jobs[job_id].pending_count == remaining
            assert keys <= connector._chunks_being_loaded
            runner.manager.complete_load.assert_not_called()

    # The completion count reaches worker_count: complete the load exactly once
    runner.manager.complete_load.assert_called_once()
    assert set(runner.manager.complete_load.call_args.args[0]) == keys
    assert job_id not in connector._jobs
    assert not keys & connector._chunks_being_loaded


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_duplicate_completion_fails_loudly(request_runner, async_scheduling):
    # The scheduler does not absorb duplicates; it fails instead of releasing
    # twice, so exactly-once delivery on the worker side is required
    runner, job_id, _ = _start_inflight_load(request_runner, async_scheduling, 2)
    connector = runner.connector_scheduler

    # More completions than workers in one delivery
    with pytest.raises(AssertionError):
        connector.update_connector_output(_worker_meta({job_id: 3}))

    # Restore pending_count, complete the job, then report the same job again
    connector._jobs[job_id].pending_count = 2
    connector.update_connector_output(_worker_meta({job_id: 2}))
    runner.manager.complete_load.assert_called_once()
    with pytest.raises(KeyError):
        connector.update_connector_output(_worker_meta({job_id: 1}))


# ---------------------------------------------------------------------------
# Scheduler side: completions for jobs dropped by reset_cache are ignored
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_stale_completion_after_reset_is_ignored(request_runner, async_scheduling):
    runner, job_id, _ = _start_inflight_load(request_runner, async_scheduling, 1)
    connector = runner.connector_scheduler

    # 1. Reset drops all job tracking while the load is unfinished
    connector.reset_cache()
    assert not connector._jobs
    assert job_id < connector._stale_job_threshold
    runner.manager.reset_mock()

    # 2. The worker's late completion arrives: it must be ignored
    connector.update_connector_output(_worker_meta({job_id: 1}))
    runner.manager.complete_load.assert_not_called()
    runner.manager.complete_store.assert_not_called()
    assert not connector._jobs

    # 3. Negative control: without the threshold the same completion raises
    connector._stale_job_threshold = 0
    with pytest.raises(KeyError):
        connector.update_connector_output(_worker_meta({job_id: 1}))
