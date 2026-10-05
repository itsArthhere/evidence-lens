import numpy as np
import pytest
from evidence_lens.retrieval import fuse, validate_query, cached_embeddings


def test_rrf_order_and_missing_candidates():
    assert fuse(['a', 'b'], ['b', 'c'], k=60, weight=.5) == ['b', 'a', 'c']


@pytest.mark.parametrize('query', ['', '   ', None, 'x' * 2001, '!!!'])
def test_bad_query(query):
    with pytest.raises(ValueError):
        validate_query(query)


def test_cache_hit_and_invalidation(tmp_path):
    calls = []
    def encode(texts):
        calls.append(texts)
        return np.tile(np.array([1, 0], dtype=np.float32), (len(texts), 1))
    a, hit = cached_embeddings(['a'], 'rev-a', 256, tmp_path, encode, expected_dimension=2)
    assert not hit
    _, hit = cached_embeddings(['a'], 'rev-a', 256, tmp_path, encode, expected_dimension=2)
    assert hit and len(calls) == 1
    _, hit = cached_embeddings(['a'], 'rev-b', 256, tmp_path, encode, expected_dimension=2)
    assert not hit and len(calls) == 2
