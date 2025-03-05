import streamlit as st 

# Sidebar options
st.sidebar.title("Chatbot Configuration")
api_key = st.sidebar.text_input("Enter API Key", type="password")
vector_db_option = st.sidebar.selectbox("Select Vector Database", ["FAISS", "Chroma"])

# Chat interface
st.title("AI Chatbot with RAG")