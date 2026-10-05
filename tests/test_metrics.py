import pytest
from evidence_lens.metrics import evaluate, rank


def test_metrics_include_missing_queries_and_all_relevant_documents():
    result = evaluate({'q': ['a'], 'extra': ['z']}, {'q': {'a': 1, 'b': 1}, 'missing': {'c': 1}})
    assert result['Recall@10'] == .25
    assert result['MRR@10'] == .5
    assert result['nDCG@10'] == pytest.approx(.5 / (1 + 1 / __import__('math').log2(3)))


def test_rank_ties_are_lexicographic_ids():
    assert rank(['2', '10', '1'], [1, 1, 0], 2) == ['10', '2']


def test_empty_judgments_rejected():
    with pytest.raises(ValueError):
        evaluate({}, {})
