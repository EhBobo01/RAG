import streamlit as st 

# Sidebar options
st.sidebar.title("Chatbot Configuration")
vector_db_option = st.sidebar.selectbox("Select Vector Database", ["FAISS", "Chroma"])
api_key = st.sidebar.text_input("Enter API Key", type="password")

# Chat interface
st.title("AI Chatbot with RAG")