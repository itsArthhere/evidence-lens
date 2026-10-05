"""Macro query metrics; missing rankings score zero, unjudged documents are nonrelevant."""
import math
import numpy as np


def rank(ids, scores, k=100):
    if k < 1 or len(ids) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('Invalid scores or cutoff')
    return [str(ids[i]) for i in np.lexsort((np.asarray(ids, dtype=str), -np.asarray(scores)))[:k]]


def evaluate(runs, qrels):
    if any(not math.isfinite(v) or v < 0 for rel in qrels.values() for v in rel.values()):
        raise ValueError('Invalid relevance grade: expected finite nonnegative values')
    if not qrels or any(not any(v > 0 for v in rel.values()) for rel in qrels.values()):
        raise ValueError('Each evaluated query needs positive judgments')
    rows = []
    for q, rel in qrels.items():
        docs = runs.get(q, [])
        if len(docs) != len(set(docs)):
            raise ValueError('Duplicate ranked document')
        relevant = {d for d, value in rel.items() if value > 0}
        hit = [int(d in relevant) for d in docs]
        dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(docs[:10]))
        ideal = sum((2 ** v - 1) / math.log2(i + 2) for i, v in enumerate(sorted(rel.values(), reverse=True)[:10]))
        rows.append([dcg / ideal, next((1 / (i + 1) for i, h in enumerate(hit[:10]) if h), 0), sum(hit[:10]) / len(relevant), sum(hit[:100]) / len(relevant)])
    return dict(zip(['nDCG@10', 'MRR@10', 'Recall@10', 'Recall@100'], np.mean(rows, axis=0).tolist()))
