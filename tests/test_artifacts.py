"""Recompute reported metrics from serialized ranked document IDs."""
import json
import os
import numpy as np
import pytest
from evidence_lens.data import ROOT, load_data, load_qrels
from evidence_lens.metrics import evaluate


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get('RUN_INTEGRATION') != '1', reason='Needs full experiment outputs and public data')
def test_serialized_rankings_reproduce_report():
    report = json.loads((ROOT / 'outputs/report.json').read_text())
    runs = json.loads((ROOT / 'outputs/test_rankings.json').read_text())
    corpus, _ = load_data()
    ids = {d['_id'] for d in corpus}
    test = load_qrels('test')
    assert len(test) == 300
    for method, run in runs.items():
        assert set(run) == set(test)
        assert all(len(ranking) == 100 and len(set(ranking)) == 100 and set(ranking) <= ids for ranking in run.values())
        actual = evaluate(run, test)
        for metric in actual:
            assert actual[metric] == pytest.approx(report['metrics'][method][metric], abs=1e-12)
    assert report['selected_hybrid']['selection_split'] == 'BEIR train'
