import streamlit as st
import os, glob
import config as cfg
import pandas as pd
from collections import Counter
from utils import ui, load_model_and_tokenenizer, get_relevant
from langchain.chat_models import ChatOpenAI
from tools.prompts import build_template
from tools.parsing import read_file,read_pkl_file
from tools.chunking import chunk_file
from tools.embedding import embed_files, get_llama_index
from tools.qa import query_folder, get_answer

st.set_page_config(page_title="ASK", layout="wide")
st.header("ASK")


folder_path = './data/*/data.pkl'
filenames = glob.glob(folder_path)
selected_filename = st.selectbox('Select a file', filenames)
st.write('You selected `%s`' % selected_filename)

with st.spinner("Indexing document... This may take a while⏳"):
    chunks = read_pkl_file(selected_filename)
    index,raw_text = get_llama_index(chunks)
    if not st.session_state.get("LOADED", ""):
        st.session_state["LOADED"] = True
        print("before loaded model", st.session_state.get("LOADED", "") )
        model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
        model, tokenizer = load_model_and_tokenenizer(model_name_or_path)
        st.session_state["MODEL"] = model
        st.session_state["TOKENIZER"] = tokenizer
        
        print("after loaded model", st.session_state.get("LOADED", "") )

with st.form(key="qa_form"):
    query = st.text_area("Ask a question about the document")
    submit = st.form_submit_button("Submit")

if submit:
    if not ui.is_query_valid(query):
        st.stop()
    # Output Columns
    answer_col, sources_col = st.columns(2)
    model = st.session_state["MODEL"]
    tokenizer = st.session_state["TOKENIZER"]
    idx = get_relevant(query, model, tokenizer, index)
    relevants = [_[0] for _ in Counter([chunks['raw_text'][i] for i in idx[0]]).most_common(5)]
    relevant_text = "\n".join(relevants)
    promt = build_template(query, relevant_text )
    result = get_answer(promt, model, tokenizer)

    with answer_col:
        st.markdown("#### Answer")
        st.markdown(result)

    with sources_col:
        st.markdown("#### Sources")
        for source in relevants:
            st.markdown(source)
            # st.markdown(source.metadata["source"])
            st.markdown("---")
