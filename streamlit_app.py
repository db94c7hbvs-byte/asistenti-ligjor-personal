import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Konfigurimi
GROQ_API_KEY = "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" 
TARGET_URL = "https://rks-gov.net" 

st.set_page_config(page_title="Asistenti Ligjor", page_icon="⚖️")
st.title("⚖️ Law AI Kosova (Personal)")

@st.cache_data(ttl=600)
def get_data(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')
    return soup.get_text()

raw_text = get_data(TARGET_URL)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_text(raw_text)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.from_texts(chunks, embeddings)
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever())

query = st.text_input("Pyetja juaj ligjore:")
if query:
    response = qa.invoke(query)
    st.info(response['result'])
