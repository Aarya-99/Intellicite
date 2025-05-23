import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from keybert import KeyBERT


# Directories

DATA_DIR = "data"
CHROMA_DIR = "chroma_store"
os.makedirs(DATA_DIR, exist_ok=True)


# Loading embedding model

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Loading the LLM

def llm():
    model_id = "google/flan-t5-base"  
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    return HuggingFacePipeline(pipeline=pipe)

llm = llm()


# Keyword extractor

def load_keybert_model():
    return KeyBERT(model="all-mpnet-base-v2")

keybert_model = load_keybert_model()


# Read PDF using PyPDF2

def read_pdf(file):
    reader = PdfReader(file)
    return "\n".join([page.extract_text() or "" for page in reader.pages])


# Storing text chunk in Chroma

def ingest_pdf(file):
    text = read_pdf(file)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = splitter.create_documents([text])
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
    vectordb.persist()
    return vectordb, text


# Load existing Chroma vector store

def get_vectorstore():
    if os.path.exists(CHROMA_DIR):
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return None


# Streamlit UI

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Intellicite: AI Assistant ChatBot for Document Analysis")

uploaded_file = st.file_uploader("Upload a PDF report:", type="pdf")
if uploaded_file:
    vectordb, full_text = ingest_pdf(uploaded_file)
    st.session_state.chat_history = []
    st.success("Document processed!")

    # Extracted keywords

    st.subheader("Keywords from the Document")
    keywords = keybert_model.extract_keywords(full_text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=10)
    kw_list = [kw[0] for kw in keywords]
    st.write(", ".join(kw_list))

    # Summary button

    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            text = full_text[:3000]
            prompt = f"Summarize the following document in 5 concise bullet points, focusing on the most important information:\n\n{text}"
            summary = llm(prompt)
            st.subheader("Summary")
            st.write(summary)


vectordb = get_vectorstore()
if vectordb:
    question = st.text_input("Ask a question about the document:")
    if question:
        with st.spinner("Finding answer..."):
            retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":5})
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            answer = qa_chain.run(question)
            st.write(answer)
