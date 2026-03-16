import streamlit as st
from rag_engine import ask_ai

st.title("Smart Node AI Assistant")

question = st.text_input("Ask a question")

if question:
    answer = ask_ai(question)
    st.write(answer)
