import streamlit as st
import os
import config as cfg
from utils import ui
from langchain.chat_models import ChatOpenAI
from tools.parsing import read_file
from tools.chunking import chunk_file
from tools.embedding import embed_files
from tools.qa import query_folder

st.set_page_config(page_title="🥳ASK", layout="wide")
st.header("Welcome to 🥳ASK")
st.subheader("ASK is a question and answer application on given documents")

st.markdown(
    "## Let's Start !\n"
    "### 1. Enter your OpenAI API key\n"
)
api_key_input = st.text_input(
    label="Paste your OpenAI API key below",
    type="password",
    value=os.environ.get("OPENAI_API_KEY", None)
          or st.session_state.get("OPENAI_API_KEY", ""),
)
openai_api_key = st.session_state.get("OPENAI_API_KEY")
if not openai_api_key:
    st.warning(
        "You can get a key at"
        " https://platform.openai.com/account/api-keys."
    )

st.markdown(
    "### 2. Choose Language\n"
)
vietnamese = st.checkbox("Vietnamese")
english = st.checkbox("English")

st.markdown(
    "### 3. Upload a pdf, docx, or txt file\n"
)

st.session_state["OPENAI_API_KEY"] = api_key_input

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


st.markdown(
    "### 4. Ask a question about the document\n"
)

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
    llm = ChatOpenAI(model=cfg.MODEL,
                     openai_api_key=openai_api_key,
                     temperature=cfg.TEMPERATURE)
    result = query_folder(
        folder_index=folder_index,
        query=query,
        return_all=False,
        llm=llm,
        is_vn=vietnamese
    )

    st.markdown("#### Answer")
    st.markdown(result.answer)

    st.markdown("#### Sources")
    for source in result.sources:
        st.markdown(source.page_content)
        st.markdown("Index: " + source.metadata["source"])
        st.markdown("---")
