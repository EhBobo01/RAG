import streamlit as st 
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from tqdm import tqdm

# Sidebar options
st.sidebar.title("Chatbot Configuration")

# Set Google API key
api_key = st.sidebar.text_input("Enter API Key", type="password")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# Embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Generative Model
gen_model = st.sidebar.selectbox("Select Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
if gen_model:
    llm = ChatGoogleGenerativeAI(model=gen_model)
    
# Vector store 
 
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)



collection = st.sidebar.file_uploader("Choose a PDF file")


# Chat interface
st.title("AI Chatbot with RAG")
st.chat_input("ask a question")