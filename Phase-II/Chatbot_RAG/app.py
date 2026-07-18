import streamlit as st
from chatbot import ask

st.title("Context-Aware RAG Chatbot")
st.caption("Ask me anything about the documents in the knowledge base.")

if "history" not in st.session_state:
    st.session_state.history = []  # list of ("human"/"ai", text) tuples

for role, message in st.session_state.history:
    with st.chat_message("user" if role == "human" else "assistant"):
        st.write(message)

user_input = st.chat_input("Ask a question...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask(user_input, st.session_state.history)
            st.write(answer)

    st.session_state.history.append(("human", user_input))
    st.session_state.history.append(("ai", answer))