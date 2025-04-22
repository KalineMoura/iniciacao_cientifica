# src/main.py
from pathlib import Path
import os
from dotenv import load_dotenv
import streamlit as st  # para @st.cache_resource

# ── env --------------------------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ── caminhos ----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "data" / "chunks_exemplos.md"


# ── embeddings + retriever --------------------------------------------------
@st.cache_resource
def get_retriever():
    from langchain_community.document_loaders import TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings

    loader = TextLoader(DATA_FILE)
    docs = loader.load()

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
    )
    vect = FAISS.from_documents(docs, embeddings)
    return vect.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 4})


# ── LLM  : Phi‑4‑mini (GGUF 4‑bit) -----------------------------------------
@st.cache_resource
def get_llm():
    from huggingface_hub import hf_hub_download
    from ctransformers import AutoModelForCausalLM

    REPO_ID = "TheBloke/phi-4-mini-instruct-GGUF"
    MODEL_FILE = "phi-4-mini-instruct.Q4_K_M.gguf"  # 4‑bit, ~0.9 GB

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILE,
        token=HF_TOKEN,
    )

    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        model_type="phi",  # importante para o tokenizer interno
        context_length=4096,
        gpu_layers=0,  # CPU‑only
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

    resposta = llm(
        prompt,
        max_new_tokens=256,
        temperature=0.1,
    )
    return resposta.strip()
