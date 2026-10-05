"""Regression coverage for retrieval/cache and evaluation review findings."""
import numpy as np
import pytest

import os
from pathlib import Path

from evidence_lens.data import fingerprint
from evidence_lens.retrieval import cached_embeddings, fuse
from evidence_lens.metrics import evaluate


def unit_vectors(rows=1, width=384):
    arr = np.zeros((rows, width), dtype=np.float32)
    arr[:, 0] = 1
    return arr


@pytest.mark.parametrize('source', ['fresh', 'cached'])
@pytest.mark.parametrize('problem', ['rows', 'width', 'rank', 'nan', 'inf', 'zero', 'norm'])
def test_invalid_embeddings_rejected(tmp_path, source, problem):
    arr = unit_vectors()
    if problem == 'rows':
        arr = unit_vectors(2)
    elif problem == 'width':
        arr = unit_vectors(width=2)
    elif problem == 'rank':
        arr = arr[0]
    elif problem == 'nan':
        arr[0, 0] = np.nan
    elif problem == 'inf':
        arr[0, 0] = np.inf
    elif problem == 'zero':
        arr[:] = 0
    elif problem == 'norm':
        arr *= 2
    path = tmp_path / (fingerprint(['a'], 'rev', 256) + '.npy')
    if source == 'cached':
        np.save(path, arr)
    with pytest.raises(ValueError, match='embedding'):
        cached_embeddings(['a'], 'rev', 256, tmp_path, lambda _: arr)
    if source == 'fresh':
        assert not path.exists()


def test_cached_float64_rejected(tmp_path):
    path = tmp_path / (fingerprint(['a'], 'rev', 256) + '.npy')
    np.save(path, unit_vectors().astype(np.float64))
    with pytest.raises(ValueError, match='embedding'):
        cached_embeddings(['a'], 'rev', 256, tmp_path, lambda _: pytest.fail('cache encoded'))


def test_fresh_float64_converted_and_valid_cache_reused(tmp_path):
    expected = unit_vectors().astype(np.float64)
    expected[0, 0] += 1e-6  # Accept normal float32 normalization roundoff.
    arr, hit = cached_embeddings(['a'], 'rev', 256, tmp_path, lambda _: expected)
    assert not hit and arr.dtype == np.float32
    cached, hit = cached_embeddings(['a'], 'rev', 256, tmp_path, lambda _: pytest.fail('cache encoded'))
    assert hit
    np.testing.assert_array_equal(arr, cached)


def test_explicit_fixture_dimension(tmp_path):
    arr, hit = cached_embeddings(['a'], 'rev', 256, tmp_path,
                                 lambda _: unit_vectors(width=2), expected_dimension=2)
    assert arr.shape == (1, 2) and not hit
    _, hit = cached_embeddings(['a'], 'rev', 256, tmp_path,
                               lambda _: pytest.fail('cache encoded'), expected_dimension=2)
    assert hit


@pytest.mark.parametrize('failure', [None, 'save', 'replace'])
def test_cache_atomic_publication_and_cleanup(tmp_path, monkeypatch, failure):
    path = tmp_path / (fingerprint(['a'], 'rev', 256) + '.npy')
    real_save, real_replace = np.save, os.replace
    published = []

    def save(file, arr, **kwargs):
        assert not path.exists(), 'final cache must not exist during write'
        real_save(file, arr, **kwargs)
        if failure == 'save':
            raise OSError('simulated save failure')

    def replace(source, destination):
        source = Path(source)
        assert source.parent == tmp_path and source != path
        assert not path.exists()
        np.testing.assert_array_equal(np.load(source), unit_vectors())
        if failure == 'replace':
            raise OSError('simulated replace failure')
        real_replace(source, destination)
        published.append(destination)

    monkeypatch.setattr(np, 'save', save)
    monkeypatch.setattr(os, 'replace', replace)
    if failure:
        with pytest.raises(OSError, match='simulated'):
            cached_embeddings(['a'], 'rev', 256, tmp_path, lambda _: unit_vectors())
        assert list(tmp_path.iterdir()) == []
    else:
        cached_embeddings(['a'], 'rev', 256, tmp_path, lambda _: unit_vectors())
        assert published == [path]
        assert list(tmp_path.iterdir()) == [path]


def test_retriever_uses_model_dimension(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import evidence_lens.retrieval as retrieval

    class Model:
        def __init__(self, *args, **kwargs):
            pass

        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, texts, **kwargs):
            return unit_vectors(len(texts), width=2)

    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=Model))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        set_num_threads=lambda _: None, cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setattr(retrieval, 'ROOT', tmp_path)
    retriever = retrieval.Retriever([{'_id': 'a', 'title': 'title', 'text': 'text'}])
    assert retriever.embeddings.shape == (1, 2)


@pytest.mark.parametrize('sparse,dense', [(['a', 'a'], ['b']), (['a'], ['b', 'b'])])
@pytest.mark.parametrize('weight', [0, .5, 1])
def test_fuse_rejects_duplicate_input_ids(sparse, dense, weight):
    with pytest.raises(ValueError, match='Duplicate'):
        fuse(sparse, dense, weight=weight)


@pytest.mark.parametrize('weight,expected', [(0, ['a']), (1, ['b'])])
def test_fuse_omits_disabled_branch_candidates(weight, expected):
    assert fuse(['a'], ['b'], weight=weight) == expected


@pytest.mark.parametrize('k', [0, -1, 1.5, float('nan'), float('inf')])
def test_fuse_rejects_invalid_k(k):
    with pytest.raises(ValueError, match='RRF'):
        fuse(['a'], ['b'], k=k)


@pytest.mark.parametrize('grade', [-1, float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('retrieved', [True, False])
def test_evaluate_rejects_invalid_grades(grade, retrieved):
    qrels = {'q': {'good': 1, 'bad': grade}}
    runs = {'q': ['good', 'bad'] if retrieved else ['good']}
    with pytest.raises(ValueError, match='relevance'):
        evaluate(runs, qrels)
