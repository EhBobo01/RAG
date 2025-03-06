import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Sidebar Inputs
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter Google API Key", type="password")
os.environ["GOOGLE_API_KEY"] = api_key

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
model_choice = st.sidebar.selectbox("Select LLM Model", ["gemini-1.5-flash", "gemini-1.5-pro"])

if uploaded_file and api_key:
    # Process PDF
    with st.spinner("Processing PDF..."):
        loader = PyPDFLoader(uploaded_file)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        splits = text_splitter.split_documents(documents)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        persist_directory = "./chroma_db"
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        for chunk in tqdm(splits, desc="Processing chunks"):
            vectorstore.add_documents([chunk], embedding=embeddings)

    # Setup LLM & QA Chain
    llm = ChatGoogleGenerativeAI(model=model_choice)
    prompt_template = """
        You are a helpful AI assistant that answers questions based on the provided PDF document.
        Use only the context provided to answer the question. If you don't know the answer or
        can't find it in the context, say so.

        Context: {context}

        Question: {question}

        Answer: Let me help you with that based on the PDF content.
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    # Main Chat Interface
    st.title("📄 PDF Chatbot")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about the document...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.spinner("Generating response..."):
            response = qa_chain(user_input)
            answer = response["result"]
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
