"""Local retrieval UI: relevance is not scientific verification."""
import json
import time
import streamlit as st
from evidence_lens.data import ROOT, load_data
from evidence_lens.retrieval import Retriever, fuse, validate_query

st.set_page_config(page_title='Evidence Lens', page_icon='🔎', layout='wide')
st.title('Evidence Lens')
st.write('Find scientific abstracts relevant to a claim. Compare lexical, neural and hybrid retrieval.')
st.info('Retrieval only: these results do not establish support, refutation, truth, or medical advice.')


@st.cache_resource
def resources():
    corpus, _ = load_data()
    return {d['_id']: d for d in corpus}, Retriever(corpus)


query = st.text_input('Scientific claim', placeholder='Vitamin D deficiency causes rickets.', max_chars=2000)
method = st.selectbox('Retrieval method', ['Hybrid RRF', 'BM25', 'MiniLM'])
if st.button('Search', type='primary'):
    try:
        query = validate_query(query)
    except ValueError as error:
        st.warning(str(error))
        st.stop()
    selection_path = ROOT / 'outputs/selection.json'
    if not selection_path.exists():
        st.error('Run python -m evidence_lens.experiment first to select the hybrid policy on development judgments.')
        st.stop()
    with st.spinner('Loading the pinned encoder and searching 5,183 abstracts…'):
        documents, retriever = resources()
        selection = json.loads(selection_path.read_text())
        start = time.perf_counter()
        sparse, dense, _ = retriever.retrieve({'query': query})
        ranking = {'BM25': sparse['query'], 'MiniLM': dense['query'],
                   'Hybrid RRF': fuse(sparse['query'], dense['query'], k=selection['k'], weight=selection['weight'])}[method]
        elapsed = time.perf_counter() - start
    st.caption(f'Warm search: {elapsed:.3f}s; includes both retrievers for comparison. Device: {retriever.device}.')
    st.caption('Source: BEIR SciFact. IDs below identify source corpus records; no paper URL mapping is assumed.')
    for position, doc_id in enumerate(ranking[:5], 1):
        doc = documents[doc_id]
        with st.expander(f"{position}. {doc['title']}", expanded=True):
            st.caption(f'BEIR SciFact corpus ID: {doc_id}')
            st.write(doc['text'])
            st.markdown('[Source corpus and citation](https://github.com/beir-cellar/beir#beers-available-datasets) · [SciFact paper](https://aclanthology.org/2020.emnlp-main.609/)')
