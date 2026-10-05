# Independent dataset audit

The BEIR archive was fetched separately from the project loader and parsed in memory with Python `requests`, `zipfile`, `json` and `csv`. TLS verification remained enabled. On this Windows environment, `urllib` could not validate the download host's certificate chain; `requests` with its certificate bundle succeeded.

## Verified archive

- Source: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
- MD5: `5f7d1de60b170fc8027bb7898e2efca1`
- SHA-256: `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`
- Corpus rows / unique document IDs: **5,183 / 5,183**
- Query rows: **1,109**
- Training/development: **809 judged query IDs, 919 positive judgment rows**
- Test: **300 judged query IDs, 339 positive judgment rows**
- Relevance grades in both judgment files: **1 only**
- Query ID overlap between train and test: **0**
- Normalized claim-text overlap between train and test: **2**

Normalization for the last check lowercases the text and collapses whitespace. Absence of ID overlap does not establish absence of content overlap. Report this overlap and the implemented handling policy alongside retrieval results; final test labels must not set fusion parameters. Exact-text checking does not establish absence of near-duplicate or upstream model-pretraining contamination.

## Judgment semantics

The original SciFact release was separately downloaded from `https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz` and parsed in memory. Its 300 development IDs exactly match BEIR's test IDs. Every BEIR test qrel set matches the corresponding original `cited_doc_ids`, including 112 claims with an empty original `evidence` annotation. Retrieval relevance therefore measures recovering citation targets, not confirming support or contradiction. The original hidden test has 300 claims with no public evidence labels.

## Independent metric check

The repository's `evaluate` implementation was checked against scikit-learn `ndcg_score` on **200 seeded random binary-relevance cases**, each with 30 candidate documents and nDCG cutoff 10. All matched within `1e-12`. This checks metric arithmetic, not the quality or completeness of the dataset judgments.

Raw corpus files and model weights are intentionally absent from Git history. Original component licensing is documented in [sources and licenses](sources-and-licenses.md).
