# src/main.py
from pathlib import Path
import os
from dotenv import load_dotenv
import streamlit as st
import time
from requests.exceptions import ReadTimeout

# ── env --------------------------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ── caminhos ----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "data" / "chunks_exemplos.md"


# ── embeddings + retriever --------------------------------------------------
@st.cache_resource
def get_retriever():
    from huggingface_hub import snapshot_download
    from langchain_community.document_loaders import TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings

    # retry + timeout elevado para o download do repo de embeddings
    for attempt in range(3):
        try:
            repo_dir = snapshot_download(
                repo_id="BAAI/bge-small-en-v1.5",
                token=HF_TOKEN,
                timeout=(60, 300),  # connect/read timeouts em segundos
            )
            break
        except ReadTimeout:
            if attempt == 2:
                raise
            time.sleep(5)  # aguarda antes de tentar de novo

    loader = TextLoader(DATA_FILE)
    docs = loader.load()

    embeddings = HuggingFaceEmbeddings(
        model_name=repo_dir,  # usa o cache local
        model_kwargs={"device": "cpu"},
    )
    vect = FAISS.from_documents(docs, embeddings)
    return vect.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 4})


# ── LLM  : Phi-3-mini (GGUF 4-bit) -----------------------------------------
@st.cache_resource
def get_llm():
    from huggingface_hub import hf_hub_download
    from ctransformers import AutoModelForCausalLM

    # retry + timeout para baixar o .gguf
    for attempt in range(3):
        try:
            model_path = hf_hub_download(
                repo_id="microsoft/phi-3-mini-4k-instruct",
                filename="Phi-3-mini-instruct-Q4_K_M.gguf",
                token=HF_TOKEN,
                timeout=(60, 600),
            )
            break
        except ReadTimeout:
            if attempt == 2:
                raise
            time.sleep(5)

    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        model_type="phi-3-mini",  # ou "phi3" conforme necessário
        context_length=4096,
        gpu_layers=0,
    )
    return llm


# ── função pública ----------------------------------------------------------
def gerar_resposta(pergunta: str) -> str:
    retriever = get_retriever()
    llm = get_llm()

    docs = retriever.get_relevant_documents(pergunta)
    contexto = "\n\n".join(d.page_content[:1000] for d in docs) or "N/D"

    prompt = f"""Você é um assistente financeiro. Com base no seguinte contexto,
responda de forma clara e objetiva. Caso não saiba, responda que não sabe.

Contexto:
{contexto}

Pergunta:
{pergunta}
"""

    resposta = llm(prompt, max_new_tokens=256, temperature=0.1)
    return resposta.strip()
