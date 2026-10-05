from evidence_lens.experiment import select_hybrid


def test_selection_uses_only_supplied_development_judgments():
    sparse = {'q': ['wrong', 'right']}
    dense = {'q': ['right', 'wrong']}
    selected, grid = select_hybrid(sparse, dense, {'q': {'right': 1}})
    assert selected['weight'] >= .5
    assert len(grid) == 9
    assert selected['selection_split'] == 'BEIR train'
