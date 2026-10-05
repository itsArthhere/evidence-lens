"""Pinned BEIR archive. Read directly from verified ZIP: extracted caches cannot drift."""
import csv
from importlib.metadata import version
import hashlib
import io
import json
from pathlib import Path
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'
MD5 = '5f7d1de60b170fc8027bb7898e2efca1'
MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


def verify_archive(path):
    if hashlib.md5(Path(path).read_bytes()).hexdigest() != MD5:
        raise ValueError('SciFact checksum mismatch; remove the corrupt archive and retry')


def archive():
    path = ROOT / 'data/scifact.zip'
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        response = requests.get(URL, timeout=120)
        response.raise_for_status()
        temp = path.with_suffix('.part')
        temp.write_bytes(response.content)
        verify_archive(temp)
        temp.replace(path)
    verify_archive(path)
    return path


def load_data():
    with zipfile.ZipFile(archive()) as z:
        corpus = [json.loads(line) for line in z.read('scifact/corpus.jsonl').decode().splitlines()]
        queries = {r['_id']: r['text'] for r in map(json.loads, z.read('scifact/queries.jsonl').decode().splitlines())}
    assert len(corpus) == 5183 and len({d['_id'] for d in corpus}) == 5183
    return corpus, queries


def load_qrels(split):
    if split not in ('train', 'test'):
        raise ValueError('Unknown split')
    with zipfile.ZipFile(archive()) as z:
        rows = csv.DictReader(io.StringIO(z.read(f'scifact/qrels/{split}.tsv').decode()), delimiter='\t')
        qrels = {}
        for row in rows:
            qrels.setdefault(row['query-id'], {})[row['corpus-id']] = int(row['score'])
    return qrels


def fingerprint(texts, revision=REVISION, max_length=256):
    payload = {'texts': texts, 'revision': revision, 'max_length': max_length, 'model': MODEL,
               'runtime': {p: version(p) for p in ['sentence-transformers', 'transformers', 'torch']},
               'policy': 'title-space-abstract;normalize=true;float32;v1'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
