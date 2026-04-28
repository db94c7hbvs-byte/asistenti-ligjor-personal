import streamlit as st
import requests
from bs4 import BeautifulSoup

# Importet e sakta
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Marrja e API Key nga Secrets
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    st.error("Gabim: API Key nuk u gjet! Shtoje te Settings > Secrets në Streamlit Cloud.")
    st.stop()

st.set_page_config(page_title="Asistenti Ligjor", page_icon="⚖️")
st.title("⚖️ Law AI Kosova")

# Funksioni për scraping
@st.cache_data(ttl=600)
def fetch_legal_content():
    try:
        url = "https://rks-gov.net"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text(separator=' ')
    except Exception as e:
        return f"Gabim gjatë marrjes së të dhënave: {e}"

# Ndërtimi i sistemit AI
with st.spinner("Duke procesuar ligjet e fundit..."):
    text_data = fetch_legal_content()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text_data])
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    llm = ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192")

    prompt = ChatPromptTemplate.from_template("""
    Je një asistent ligjor profesional për ligjet e Kosovës. 
    Përgjigju vetëm në gjuhën shqipe duke u bazuar në këtë kontekst:
    <context>
    {context}
    </context>
    Pyetja: {input}
    """)

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

# Interface
query = st.text_input("Pyetni diçka për ligjet e fundit (Gazeta Zyrtare):")

if query:
    with st.spinner("Duke kërkuar përgjigjen..."):
        try:
            response = retrieval_chain.invoke({"input": query})
            st.markdown("### Përgjigjja:")
            st.info(response["answer"])
        except Exception as e:
            st.error(f"Ndodhi një gabim: {e}")
