"""Train-qrels-only selection, then one held-out evaluation of all 300 queries."""
import json
import platform
import time
from importlib.metadata import version
import pandas as pd
from .data import ROOT, MD5, MODEL, REVISION, load_data, load_qrels
from .metrics import evaluate
from .retrieval import Retriever, fuse, tokenize


def select_hybrid(sparse, dense, development):
    grid = []
    for k in (10, 30, 60):
        for weight in (.25, .5, .75):
            runs = {q: fuse(sparse[q], dense[q], k=k, weight=weight) for q in development}
            grid.append({'k': k, 'weight': weight, **evaluate(runs, development)})
    # Stable grid order is the preregistered tie policy (lower k, then lower dense weight).
    best = max(grid, key=lambda r: r['nDCG@10'])
    return {'k': best['k'], 'weight': best['weight'], 'selection_split': 'BEIR train', 'objective': 'nDCG@10', 'candidate_depth': 100}, grid


def run():
    out = ROOT / 'outputs'
    out.mkdir(exist_ok=True)
    corpus, queries = load_data()
    train = load_qrels('train')
    assert all(d in {c['_id'] for c in corpus} for rel in train.values() for d in rel)
    retriever = Retriever(corpus)
    sparse, dense, development_timing = retriever.retrieve({q: queries[q] for q in train})
    selected, grid = select_hybrid(sparse, dense, train)
    # Persist the frozen decision before loading test qrels or retrieving test queries.
    (out / 'selection.json').write_text(json.dumps(selected, indent=2))
    pd.DataFrame(grid).to_csv(out / 'development_grid.csv', index=False)
    test = load_qrels('test')
    assert len(test) == 300 and not set(test) & set(train)
    sparse, dense, test_timing = retriever.retrieve({q: queries[q] for q in test})
    start = time.perf_counter()
    hybrid = {q: fuse(sparse[q], dense[q], k=selected['k'], weight=selected['weight']) for q in test}
    test_timing['rrf_fusion_seconds'] = time.perf_counter() - start
    runs = {'BM25': sparse, 'MiniLM': dense, 'Hybrid RRF': hybrid}
    metrics = {name: evaluate(run, test) for name, run in runs.items()}
    pd.DataFrame(metrics).T.rename_axis('method').to_csv(out / 'metrics.csv')
    rows = []
    for q, rel in test.items():
        row = {'query_id': q, 'claim': queries[q], 'relevant_ids': '|'.join(rel)}
        for name, run in runs.items():
            row[name + '_nDCG@10'] = evaluate({q: run[q]}, {q: rel})['nDCG@10']
            row[name + '_top1'] = run[q][0]
            row[name + '_first_relevant_rank'] = next((i for i, d in enumerate(run[q], 1) if d in rel), None)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / 'per_query.csv', index=False)
    (out / 'test_rankings.json').write_text(json.dumps(runs))
    # Post-selection diagnostics do not change the frozen policy.
    normalize = lambda text: ' '.join(tokenize(text))
    train_texts = {normalize(queries[q]) for q in train}
    overlapping = [q for q in test if normalize(queries[q]) in train_texts]
    clean_test = {q: rel for q, rel in test.items() if q not in overlapping}
    token_lengths = [len(x) for x in retriever.model.tokenizer(retriever.texts, truncation=False, verbose=False)['input_ids']]
    audit = {'corpus_documents': len(corpus), 'queries_total': len(queries), 'development_queries': len(train), 'test_queries': len(test),
             'train_test_normalized_text_overlap': overlapping, 'normalization': 'lowercase word tokens joined by spaces',
             'empty_titles': sum(not d['title'].strip() for d in corpus), 'empty_abstracts': sum(not d['text'].strip() for d in corpus),
             'duplicate_corpus_texts': len(corpus) - len({normalize(t) for t in retriever.texts}),
             'max_sequence_length': 256, 'documents_truncated': sum(n > 256 for n in token_lengths),
             'wordpiece_length_quantiles': pd.Series(token_lengths).quantile([0, .5, .9, .99, 1]).to_dict(),
             'test_relevant_documents_per_query': pd.Series([len(r) for r in test.values()]).value_counts().to_dict(),
             'nonoverlap_sensitivity': {name: evaluate(run, clean_test) for name, run in runs.items()} if clean_test else {}}
    pd.DataFrame({'wordpieces': token_lengths, 'abstract_words': [len(d['text'].split()) for d in corpus]}).to_csv(out / 'document_lengths.csv', index=False)
    report = {'metrics': metrics, 'selected_hybrid': selected, 'audit': audit,
              'timings': {'index': retriever.timings, 'development': development_timing, 'test': test_timing},
              'provenance': {'dataset_md5': MD5, 'model': MODEL, 'revision': REVISION, 'embedding_cache_hit': retriever.cache_hit,
                             'device': retriever.device, 'torch_threads': 4, 'platform': platform.platform(), 'python': platform.python_version(),
                             'packages': {p: version(p) for p in ['torch', 'sentence-transformers', 'numpy', 'rank-bm25', 'streamlit']}}}
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    run()
