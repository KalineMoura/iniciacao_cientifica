from pathlib import Path
import os
from dotenv import load_dotenv
import streamlit as st

# ── env & caminhos ----------------------------------------------------------
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "data" / "chunks_exemplos.md"
MODEL_FILE = BASE_DIR.parent / "models" / "phi3-mini.q4.gguf"   # ← caminho local

# ── embeddings + retriever (igual ao anterior) ------------------------------
@st.cache_resource
def get_retriever():
    from langchain_community.document_loaders import TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from huggingface_hub import snapshot_download

    repo_dir = snapshot_download("BAAI/bge-small-en-v1.5")
    docs = TextLoader(DATA_FILE).load()
    emb = HuggingFaceEmbeddings(model_name=repo_dir, model_kwargs={"device": "cpu"})
    vect = FAISS.from_documents(docs, emb)
    return vect.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 4})

# ── LLM via ctransformers ----------------------------------------------------
@st.cache_resource
def get_llm():
    from ctransformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_FILE,
        model_type="phi",         # backend já conhece a arquitetura
        gpu_layers=0              # 0 → tudo em CPU
    )
    return model

# ── geração -----------------------------------------------------------------
def gerar_resposta(pergunta: str) -> str:
    retriever = get_retriever()
    model     = get_llm()

    docs = retriever.get_relevant_documents(pergunta)
    contexto = "\n\n".join(d.page_content[:1000] for d in docs) or "N/D"

    prompt = f"""
Você é um assistente financeiro. Com base no seguinte contexto,
responda de forma clara e objetiva. Caso não saiba, responda que não sabe.

Contexto:
{contexto}

Pergunta:
{pergunta}

Resposta:
"""
    # ctransformers usa chamada direta ao modelo (“pipeline” inline)
    generated = model(
        prompt,
        max_new_tokens=256,
        temperature=0.1,
        stream=False        # True se quiser streaming palavra a palavra
    )
    return generated.strip()
