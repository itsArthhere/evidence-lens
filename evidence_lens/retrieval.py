"""Exact sparse/dense retrieval and development-selected reciprocal rank fusion."""
import os
import re
import tempfile
import time
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from .data import ROOT, MODEL, REVISION, fingerprint
from .metrics import rank


def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


def validate_query(query):
    if not isinstance(query, str) or not query.strip() or len(query) > 2000 or not tokenize(query):
        raise ValueError('Enter a claim with words (1–2000 characters).')
    return query.strip()


def fuse(sparse, dense, k=60, weight=.5, cutoff=100):
    if not np.isfinite(k) or k <= 0 or k != int(k) or not 0 <= weight <= 1:
        raise ValueError('Invalid RRF parameters')
    scores = {}
    for docs, w in [(sparse, 1 - weight), (dense, weight)]:
        if len(docs) != len(set(docs)):
            raise ValueError('Duplicate ranked document in fusion input')
        if w == 0:
            continue
        for i, doc in enumerate(docs, 1):
            scores[doc] = scores.get(doc, 0) + w / (k + i)
    return rank(list(scores), list(scores.values()), cutoff)


def cached_embeddings(texts, revision, max_length, directory, encode, expected_dimension=384):
    """Load normalized float32 vectors; convert fresh encoder output to float32."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (fingerprint(texts, revision, max_length) + '.npy')
    hit = path.exists()
    arr = np.load(path, allow_pickle=False) if hit else np.asarray(encode(texts), dtype=np.float32)
    if (arr.shape != (len(texts), expected_dimension)
            or arr.dtype != np.float32 or not np.isfinite(arr).all()
            or not np.allclose(np.linalg.norm(arr.astype(np.float64), axis=1), 1, rtol=0, atol=1e-5)):
        raise ValueError('Invalid embedding array: expected normalized finite float32 vectors of the model dimension')
    if not hit:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, suffix='.tmp', delete=False) as file:
                temporary = Path(file.name)
                np.save(file, arr, allow_pickle=False)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return arr, hit


class Retriever:
    def __init__(self, corpus):
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(4)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ids = [d['_id'] for d in corpus]
        self.texts = [(d['title'] + ' ' + d['text']).strip() for d in corpus]
        self.timings = {}
        start = time.perf_counter()
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        self.timings['bm25_build_seconds'] = time.perf_counter() - start
        start = time.perf_counter()
        self.model = SentenceTransformer(MODEL, revision=REVISION, device=self.device)
        self.model.max_seq_length = 256
        self.timings['model_load_seconds'] = time.perf_counter() - start
        start = time.perf_counter()
        self.embeddings, self.cache_hit = cached_embeddings(
            self.texts, REVISION, 256, ROOT / '.cache/embeddings', self.encode,
            expected_dimension=self.model.get_sentence_embedding_dimension())
        self.timings['corpus_encode_or_cache_seconds'] = time.perf_counter() - start

    def encode(self, texts):
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    def retrieve(self, queries):
        texts = [validate_query(t) for t in queries.values()]
        start = time.perf_counter()
        sparse = {q: rank(self.ids, self.bm25.get_scores(tokenize(t))) for q, t in zip(queries, texts)}
        sparse_seconds = time.perf_counter() - start
        start = time.perf_counter()
        vectors = self.encode(texts)
        scores = vectors @ self.embeddings.T
        dense = {q: rank(self.ids, s) for q, s in zip(queries, scores)}
        dense_seconds = time.perf_counter() - start
        return sparse, dense, {'bm25_seconds': sparse_seconds, 'dense_encode_and_search_seconds': dense_seconds, 'query_count': len(texts), 'mode': 'batched throughput, not single-request latency'}
