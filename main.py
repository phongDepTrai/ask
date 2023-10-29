import streamlit as st
import os
import config as cfg
from utils import ui
from langchain.chat_models import ChatOpenAI
from tools.parsing import read_file
from tools.chunking import chunk_file
from tools.embedding import embed_files
from tools.qa import query_folder

st.set_page_config(page_title="ASK", layout="wide")
st.header("ASK")

with st.sidebar:
    st.markdown(
        "## How to use\n"
        "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑\n"  # noqa: E501
        "2. Upload a pdf, docx, or txt file📄\n"
        "3. Ask a question about the document💬\n"
    )
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="Paste your OpenAI API key here (sk-...)",
        help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
        value=os.environ.get("OPENAI_API_KEY", None)
              or st.session_state.get("OPENAI_API_KEY", ""),
    )

    st.session_state["OPENAI_API_KEY"] = api_key_input

openai_api_key = st.session_state.get("OPENAI_API_KEY")
if not openai_api_key:
    st.warning(
        "Enter your OpenAI API key in the sidebar. You can get a key at"
        " https://platform.openai.com/account/api-keys."
    )

uploaded_file = st.file_uploader(
    "Upload a pdf, docx, or txt file",
    type=["pdf", "docx", "txt"],
    help="Scanned documents are not supported yet!",
)

if not uploaded_file:
    st.stop()

file = read_file(uploaded_file)


chunked_file = chunk_file(file,
                          chunk_size=cfg.CHUNK_SIZE,
                          chunk_overlap=cfg.CHUNK_OVERLAP)

if not ui.is_open_ai_key_valid(openai_api_key):
    st.stop()

with st.spinner("Indexing document... This may take a while⏳"):
    folder_index = embed_files(
        files=[chunked_file],
        embedding=cfg.EMBEDDING,
        vector_store=cfg.VECTOR_STORE,
        openai_api_key=openai_api_key,
    )


with st.form(key="qa_form"):
    query = st.text_area("Ask a question about the document")
    submit = st.form_submit_button("Submit")

if submit:
    if not ui.is_query_valid(query):
        st.stop()

    # Output Columns
    answer_col, sources_col = st.columns(2)
    llm = ChatOpenAI(model=cfg.MODEL,
                     openai_api_key=openai_api_key,
                     temperature=cfg.TEMPERATURE)
    result = query_folder(
        folder_index=folder_index,
        query=query,
        return_all=False,
        llm=llm,
    )

    with answer_col:
        st.markdown("#### Answer")
        st.markdown(result.answer)

    with sources_col:
        st.markdown("#### Sources")
        for source in result.sources:
            st.markdown(source.page_content)
            st.markdown(source.metadata["source"])
            st.markdown("---")
