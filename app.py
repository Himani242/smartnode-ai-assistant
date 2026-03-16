import streamlit as st
from rag_engine import ask_ai

st.title("Smart Node AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Ask something")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    answer = ask_ai(user_input)
    st.session_state.messages.append({"role": "assistant", "content": answer})

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
