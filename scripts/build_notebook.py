"""Generate a readable, stepwise notebook; execute separately with execute_notebook.py."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    compile(text, '<notebook-cell>', 'exec')
    cells.append(nbf.v4.new_code_cell(text))

md('''# Evidence Lens: scientific abstract retrieval

Which retriever finds relevant scientific abstracts: lexical BM25, a pretrained neural encoder, or their combination?

This notebook evaluates all **5,183 documents** and all **300 BEIR SciFact test queries**. It is document retrieval, not entailment, claim verification, an LLM answer generator, or medical advice. The encoder is pretrained; no fine-tuning happens here.

We use BEIR train judgments only to select hybrid parameters. BEIR test corresponds to the original SciFact labeled development set, **not the original hidden test**. Test diagnostics are descriptive and never feed selection.''')
md('''## 1. Sources and protocol

[SciFact, Wadden et al. (2020)](https://aclanthology.org/2020.emnlp-main.609/) supplies claims and abstracts; [BEIR](https://github.com/beir-cellar/beir) packages document-relevance judgments. Claims/evidence: CC BY 4.0; original abstracts: ODC-By 1.0; BEIR dataset card: CC BY-SA 4.0. See [licensing notes](../docs/sources-and-licenses.md).

The archive is verified against BEIR's published MD5 on every read. MiniLM's actual Hugging Face revision is pinned. Corpus vectors are cached by full input text, model revision and preprocessing policy. MD5 is an upstream integrity check, not a cryptographic authenticity guarantee.''')
code('''from pathlib import Path
import sys, json
ROOT = Path.cwd() if (Path.cwd() / 'evidence_lens').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from evidence_lens.data import load_data, load_qrels, MODEL, REVISION, MD5
from evidence_lens.retrieval import Retriever, fuse, tokenize
from evidence_lens.metrics import evaluate
from evidence_lens.experiment import select_hybrid
np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')
print({'model': MODEL, 'revision': REVISION, 'archive_md5': MD5})''')
md('''## 2. Load the benchmark and inspect the development data

`corpus` has `_id`, `title`, `text`, and metadata. `queries` maps claim IDs to text. Qrels map query IDs to relevant corpus IDs with positive relevance grades. The corpus is the fixed search collection, so BM25's document-frequency statistics legitimately use the entire collection. No labels enter either retriever.

We inspect development judgments before choosing the hybrid. Merely loading the unlabeled query mapping does not select on test outcomes.''')
code('''corpus, queries = load_data()
development = load_qrels('train')
assert len(corpus) == 5183
assert all(q in queries for q in development)
corpus_ids = {d['_id'] for d in corpus}
assert all(d in corpus_ids for rel in development.values() for d in rel)
quality = {'documents': len(corpus), 'all_queries': len(queries), 'development_queries': len(development),
           'missing_titles': sum(not d['title'].strip() for d in corpus),
           'missing_abstracts': sum(not d['text'].strip() for d in corpus),
           'duplicate_ids': len(corpus) - len(corpus_ids)}
display(pd.Series(quality, name='count'))
display(pd.Series([len(r) for r in development.values()]).value_counts().rename('development queries by relevant-document count'))
display(pd.DataFrame(corpus)[['_id', 'title']].head(3))''')
code('''word_lengths = pd.Series([len(d['text'].split()) for d in corpus])
display(word_lengths.describe(percentiles=[.5, .9, .99]).to_frame('abstract words'))
fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(word_lengths, bins=50, color='#247c8e')
ax.set(xlabel='Abstract word count', ylabel='Documents', title='Full scientific abstracts vary substantially in length')
fig.tight_layout()
(ROOT / 'outputs/figures').mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / 'outputs/figures/abstract_lengths.png', dpi=150)
plt.show()''')
md('''## 3. Baseline and neural representations

**BM25:** lowercase Unicode word tokens; `rank_bm25` defaults `k1=1.5`, `b=0.75`; no stemming or stop-word removal. This preserves scientific terms and provides an auditable lexical baseline.

**MiniLM:** title + space + full abstract, encoded to normalized 384-dimensional vectors. Dot product is cosine similarity. Inputs are truncated to **256 wordpieces**, including special tokens; the full abstract remains visible in the app. We measure truncation rather than implying that the encoder reads every abstract in full. No chunking, training or domain adaptation is implemented.

Exact search is adequate at 5,183 documents. Ties resolve by lexicographic corpus ID. Four CPU torch threads are used unless CUDA is available. Timings separate index/model setup from batched query encoding/search; they are not production latency guarantees.''')
code('''retriever = Retriever(corpus)
print({'device': retriever.device, 'embedding_shape': retriever.embeddings.shape,
       'embedding_cache_hit': retriever.cache_hit, **retriever.timings})
lengths = [len(x) for x in retriever.model.tokenizer(retriever.texts, truncation=False, verbose=False)['input_ids']]
print({'truncated_documents': sum(n > 256 for n in lengths), 'total_documents': len(lengths), 'max_input_wordpieces': max(lengths)})''')
md('''## 4. Select hybrid fusion using development judgments only

We retrieve the top 100 candidates from each method and fuse their ranks:

$$s(d) = \\frac{1-w}{k+r_{BM25}(d)} + \\frac{w}{k+r_{dense}(d)}.$$

Missing candidates contribute zero. Search grid: `k ∈ {10,30,60}`, dense weight `w ∈ {0.25,0.5,0.75}`. Select highest development nDCG@10; exact ties choose lower k, then lower dense weight. These choices were fixed before held-out evaluation.

All metrics are macro-averages over qrels queries, including queries with no retrieved positives. nDCG uses gain `2^relevance−1`; MRR@10 is reciprocal first relevant rank or zero. Recall's denominator is **all judged relevant documents**, not retrieved positives. Unjudged results count as nonrelevant; incomplete judgments remain a limitation.''')
code('''dev_sparse, dev_dense, dev_timing = retriever.retrieve({q: queries[q] for q in development})
selected, grid = select_hybrid(dev_sparse, dev_dense, development)
display(pd.DataFrame(grid))
print('Frozen selection:', selected)
# This file is written before test judgments are opened in this notebook.
(ROOT / 'outputs/notebook_selection.json').write_text(json.dumps(selected, indent=2))
print(dev_timing)''')
md('''## 5. Open held-out judgments only after freezing selection

This is a reproducibility rerun of the same frozen experiment, not another tuning opportunity. We report all three predeclared methods on all 300 queries. Re-execution must match the committed selection and metric report within numerical tolerance.''')
code('''test = load_qrels('test')
assert len(test) == 300 and not set(test) & set(development)
sparse, dense, test_timing = retriever.retrieve({q: queries[q] for q in test})
hybrid = {q: fuse(sparse[q], dense[q], k=selected['k'], weight=selected['weight']) for q in test}
runs = {'BM25': sparse, 'MiniLM': dense, 'Hybrid RRF': hybrid}
comparison = pd.DataFrame({name: evaluate(run, test) for name, run in runs.items()}).T
display(comparison.round(4))
print(test_timing)
report = json.loads((ROOT / 'outputs/report.json').read_text())
assert selected == report['selected_hybrid']
for name in runs:
    np.testing.assert_allclose(comparison.loc[name].values, list(report['metrics'][name].values()), atol=1e-8)
print('Verified: notebook rankings reproduce the pipeline metrics.')''')
code('''fig, ax = plt.subplots(figsize=(9, 4))
comparison.plot.bar(ax=ax, rot=0, color=['#235789', '#27a69a', '#e5b45a', '#c86a57'])
ax.set(ylabel='Macro score', ylim=(0, 1), title='BEIR SciFact · 300 held-out claims · 5,183 documents')
ax.legend(loc='lower right')
fig.tight_layout()
fig.savefig(ROOT / 'outputs/figures/retrieval_comparison.png', dpi=150)
plt.show()
best = comparison['nDCG@10'].idxmax()
print(f'Highest observed test nDCG@10: {best}. This observation does not retune the deployed hybrid.')''')
md('''## 6. Leakage, denominator and length audit

Official splits are retained. We normalize claims to lowercase word-token sequences and check cross-split duplicates after policy selection. A non-overlap sensitivity report is recorded in the pipeline audit. Query ID disjointness does not establish pretraining decontamination: the encoder's upstream training includes scientific citation pairs and may overlap this benchmark. This project makes no clean-room generalization claim.''')
code('''audit = report['audit']
display(pd.Series({k: v for k, v in audit.items() if k not in ['nonoverlap_sensitivity', 'wordpiece_length_quantiles']}, dtype=object))
display(pd.DataFrame(audit['nonoverlap_sensitivity']).T)
print('Full pipeline timing breakdown:')
display(report['timings'])''')
md('''## 7. Error analysis: wins, regressions and complete misses

Select cases deterministically by per-query nDCG delta (hybrid minus BM25), not by manually choosing favorable anecdotes. A retrieved paper can share scientific vocabulary yet fail on population, intervention, direction of effect or evidence type. Relevance judgments here cannot certify the truth of the claim.''')
code('''rows = []
for q, rel in test.items():
    row = {'query_id': q, 'claim': queries[q]}
    for name, run in runs.items():
        row[name] = evaluate({q: run[q]}, {q: rel})['nDCG@10']
    row['hybrid_minus_BM25'] = row['Hybrid RRF'] - row['BM25']
    rows.append(row)
errors = pd.DataFrame(rows).sort_values(['hybrid_minus_BM25', 'query_id'])
display(errors.head(3))
display(errors.tail(3))
print('Hybrid zero-nDCG@10 claims:', int((errors['Hybrid RRF'] == 0).sum()))
lookup = {d['_id']: d for d in corpus}
for q in [errors.iloc[0]['query_id'], errors.iloc[-1]['query_id']]:
    print('\\nCLAIM:', queries[q])
    print('JUDGED RELEVANT:', [(d, lookup[d]['title']) for d in test[q]])
    print('TOP RESULT BY METHOD:', {name: (run[q][0], lookup[run[q][0]]['title']) for name, run in runs.items()})''')
md('''## 8. Practical interpretation and limits

The hybrid combines independent lexical and pretrained neural rankings without training a new model. Its parameters are chosen on BEIR train judgments. The observed comparison above—not an assumption that neural is always better—determines the scientific conclusion.

- Small, English biomedical/scientific benchmark; no current-literature coverage guarantee.
- Abstract retrieval only; no full-paper ingestion, citation verification, support/refutation classification or generated answers.
- Truncation loses late-abstract details. Chunking and a domain encoder are untested future alternatives, not implemented features.
- Incomplete qrels penalize unjudged but possibly relevant documents. No calibrated confidence or relevance threshold is provided; every valid query receives ranked candidates.
- No repeated hardware measurements, uncertainty intervals, production load tests or statistical-significance claims.
- The local Streamlit app shows title, full source abstract and corpus ID. Dataset/paper citation links are provided; document URLs are deliberately not guessed.

## 9. Practices checklist

| Practice | Status / rationale |
|---|---|
| Real full dataset | 5,183 verified corpus records; full 809 development and 300 test queries |
| Data integrity | Published MD5 checked on every cached ZIP read |
| Baseline | Token BM25 before pretrained MiniLM and RRF |
| Held-out selection | Nine RRF settings selected using train qrels only |
| Scaling / imputation | Not applicable to text retrieval; no numeric feature fitting |
| Fine-tuning | Not performed; do not interpret this as a trained SciFact model |
| Cache reproducibility | Text/model/policy fingerprint and pinned revision |
| Leakage audit | Query IDs and normalized claim overlap; pretraining overlap unknown |
| Tests | Metric denominators, ties, invalid input, cache changes, real Streamlit AppTest |
| Runtime verification | Full pipeline and this notebook executed; recorded results checked |
| Deployment | Local Streamlit; no auth, multi-user service, or production monitoring |

See the [README](../README.md) for commands and [system design](../docs/design.md) for implementation boundaries.''')
nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
path = root / 'notebooks/01_evidence_retrieval.ipynb'
path.parent.mkdir(exist_ok=True)
nbf.write(nb, path)
print(path)
