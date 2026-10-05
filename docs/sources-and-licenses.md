# Sources and licensing

## SciFact, not production clinical data

EvidenceLens uses the BEIR distribution of SciFact, the scientific-claim benchmark from Wadden et al., [Fact or Fiction: Verifying Scientific Claims](https://aclanthology.org/2020.emnlp-main.609/) (EMNLP 2020). Claims are expert-written benchmark examples associated with scientific abstracts; they are not a feed of current clinical recommendations.

- Original project: https://github.com/allenai/scifact
- Original data schema: https://github.com/allenai/scifact/blob/master/doc/data.md
- Original license: https://github.com/allenai/scifact/blob/master/LICENSE.md
- BEIR dataset card: https://huggingface.co/datasets/BeIR/scifact
- BEIR download: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip

The original license distinguishes components:

| Component | Original license |
|---|---|
| Claims and evidence annotations | CC BY 4.0 |
| Corpus abstracts, sourced from S2ORC | ODC-By 1.0 |
| Original SciFact code | Apache-2.0 |

The BEIR Hugging Face dataset card separately lists **CC BY-SA 4.0** for the packaged dataset. That metadata does not make the source abstracts public domain or replace the component-specific attribution above. Preserve the original and redistribution notices for reused material; this repository makes no independent commercial-use determination. Raw corpus files are downloaded locally rather than committed. Any displayed claim/abstract examples originate from this benchmark, not from newly authored scientific assertions.

## Evaluation naming

The original SciFact hidden test labels are not public. BEIR's 300 labeled retrieval test queries are the original development claims, not that hidden test. Report results as **BEIR SciFact retrieval evaluation**, not as a hidden-test claim-verification score.

An independent comparison found that **all 300 BEIR test qrel sets equal the original claim's `cited_doc_ids`**. These are source-citation targets, not exclusively abstracts annotated to support or contradict the claim. **112 of those 300 original claims have no annotated support/contradiction evidence**, although each still has retrieval targets. Consequently, a successful retrieval score means recovering a judged citation target, not finding proof. The retrieval qrels do not provide a support-versus-contradiction verdict for the application to learn.

## Sentence encoder

Model: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

The model card lists Apache-2.0, 384-dimensional embeddings, and default truncation beyond 256 word pieces. It is a pretrained general sentence encoder, not a SciFact-fine-tuned verifier. The training corpus includes scientific citation pairs; this project cannot establish that its upstream pretraining data was free of overlap with the benchmark. Pinning a model revision provides reproducibility, not proof of pretraining decontamination.

Model weights are downloaded from the model publisher. This repository does not redistribute them or claim to have trained the encoder.
