"""Check that reading completion metadata consumes it exactly once."""

import pytest

from tests.v1.kv_connector.unit.offloading_connector.test_worker import (
    _load_metadata,
    _make_worker,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import TransferResult


@pytest.mark.parametrize("batches", [[[10, 11]], [[10], [11]], [[], [11], [], [10]]])
@pytest.mark.parametrize("read_each_batch", [False, True])
def test_completion_is_reported_once(batches, read_each_batch):
    connector, _ = _make_worker(
        KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[])
    )
    for job_id in (10, 11):
        connector.start_kv_transfers(_load_metadata(job_id))

    delivered = []
    for batch in batches:
        connector.worker.get_finished.return_value = [
            TransferResult(job_id, True) for job_id in batch
        ]
        sent, received = connector.get_finished(set())
        assert sent == set()
        assert received == ({"req"} if batch else set())

        if read_each_batch:
            metadata = connector.build_connector_worker_meta()
            if metadata is not None:
                assert metadata.completed_jobs == {job_id: 1 for job_id in batch}
                delivered.extend(metadata.completed_jobs)
            else:
                assert not batch
            assert connector.build_connector_worker_meta() is None

    if not read_each_batch:
        metadata = connector.build_connector_worker_meta()
        assert metadata.completed_jobs == {10: 1, 11: 1}
        delivered.extend(metadata.completed_jobs)

    assert sorted(delivered) == [10, 11]
    assert not connector._load_jobs
    connector.worker.get_finished.return_value = []
    assert connector.get_finished(set()) == (set(), set())
    assert connector.build_connector_worker_meta() is None
