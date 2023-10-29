import streamlit as st


def is_query_valid(query):
    if not query:
        st.error("Please enter a question!")
        return False
    return True


@st.cache_data(show_spinner=False)
def is_open_ai_key_valid(openai_api_key):
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar!")
        return False
    return True
