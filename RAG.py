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

def main():
    st.title("📄 AI Chatbot for PDFs")
    
    # Input API Key
    api_key = st.text_input("Enter your Google API Key:", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    else:
        st.warning("Please enter a valid API key to proceed.")
        return
    
    # Model Selection
    model_option = st.selectbox("Select a model:", ["gemini-1.5-flash", "gemini-1.5-pro"])
    
    # Upload PDF
    uploaded_file = st.file_uploader("Upload a PDF file:", type=["pdf"])
    if uploaded_file is None:
        st.info("Please upload a PDF to proceed.")
        return
    
    # Load PDF and Split Text
    with st.spinner("Processing PDF..."):
        loader = PyPDFLoader(uploaded_file)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        splits = text_splitter.split_documents(documents)
    
    # Generate Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    persist_directory = "./chroma_db"
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    # Store Document Chunks
    for chunk in tqdm(splits, desc="Processing chunks"):
        vectorstore.add_documents([chunk], embedding=embeddings)
    
    # Set up LLM and QA Chain
    llm = ChatGoogleGenerativeAI(model=model_option)
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
    
    # Chat Interface
    query = st.text_input("Ask a question about the document:")
    if st.button("Submit") and query:
        with st.spinner("Fetching answer..."):
            result = qa_chain.run(query)
            st.write("\n💡 **Answer:**\n", result["result"])
            
            st.write("\n📚 **Source Documents:**")
            for doc in result["source_documents"]:
                st.write(f"🔹 {doc.metadata['source']} -> {doc.page_content[:200]}...")

if __name__ == "__main__":
    main()
