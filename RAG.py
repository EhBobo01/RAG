import streamlit as st 

# Sidebar options
st.sidebar.title("Chatbot Configuration")
api_key = st.sidebar.text_input("Enter API Key", type="password")
llm = st.sidebar.selectbox("Select Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
collection = st.sidebar.file_uploader("Choose a file")


# Chat interface
st.title("AI Chatbot with RAG")
st.chat_input("ask a question")