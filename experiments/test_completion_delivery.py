import pytest

from tests.v1.kv_connector.unit.offloading_connector.test_worker import (
    _load_metadata,
    _make_worker,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import TransferResult


# Completion arrival: all at once / one per step / empty steps in reverse order
@pytest.mark.parametrize("batches", [[[10, 11]], [[10], [11]], [[], [11], [], [10]]])
# Metadata reading: once at the end / after every step
@pytest.mark.parametrize("read_each_batch", [False, True])
def test_completion_is_reported_once(batches, read_each_batch):
    # 1. Create a worker connector with a mocked transfer engine (no GPU)
    connector, _ = _make_worker(
        KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[])
    )
    # 2. Start load jobs 10 and 11 for the same request
    for job_id in (10, 11):
        connector.start_kv_transfers(_load_metadata(job_id))

    delivered = []
    for batch in batches:
        # 3. Inject this step's completions and collect them
        connector.worker.get_finished.return_value = [
            TransferResult(job_id, True) for job_id in batch
        ]
        sent, received = connector.get_finished(set())
        assert sent == set()
        assert received == ({"req"} if batch else set())

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

    # 5. Each job is delivered once and no state remains
    assert sorted(delivered) == [10, 11]
    assert not connector._load_jobs
    connector.worker.get_finished.return_value = []
    assert connector.get_finished(set()) == (set(), set())
    assert connector.build_connector_worker_meta() is None
