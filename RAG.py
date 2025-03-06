__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Streamlit App Title
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📖 RAG-Powered Chatbot")

# Settings Sidebar
st.sidebar.header("⚙️ Settings")
api_key = st.sidebar.text_input("Enter Google API Key", type="password")
#api_key = "AIzaSyDEUAl1yUDjQ_f0k6CzFycSe_jd55R-bdk"

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    st.sidebar.success("API Key Set!")
else:
    st.sidebar.warning("Please enter your API key to proceed.")

# Load ChromaDB Vector Store
persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Custom Prompt Template
prompt_template = """
You are an AI assistant answering questions based on a PDF document.
Use only the provided context to generate answers. If uncertain, state so.

Context: {context}

Question: {question}

Answer: """

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Create QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# Chat Interface
st.subheader("💬 Chat with the AI")

# Initialize chat history if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
user_query = st.chat_input("Ask a question:")

if user_query:
    if not api_key:
        st.error("Please enter your API key in the settings.")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        
        with st.spinner("Fetching answer..."):
            result = qa_chain(user_query)
            model_answer = result['result']
            
            # Display AI response
            st.session_state.messages.append({"role": "assistant", "content": model_answer})
            with st.chat_message("assistant"):
                st.markdown(model_answer)
            
            # Display Sources
            with st.expander("📚 Source Documents"):
                for doc in result["source_documents"]:
                    st.markdown(f"🔹 `{doc.metadata['source']}` - {doc.page_content[:200]}...")
