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
    # ↓ snapshot_download baixa (e cacheia) o repo completo localmente
    from huggingface_hub import snapshot_download

    repo_dir = snapshot_download(repo_id="BAAI/bge-small-en-v1.5", token=HF_TOKEN)

    # carregando os documentos
    from langchain_community.document_loaders import TextLoader

    loader = TextLoader(DATA_FILE)
    docs = loader.load()

    # inicializando embeddings a partir da pasta local (repo_dir contém config.json)
    from langchain.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=repo_dir, model_kwargs={"device": "cpu"}
    )

    # criando o FAISS
    from langchain_community.vectorstores import FAISS

    vect = FAISS.from_documents(docs, embeddings)
    return vect.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 4})


# ── LLM  : Phi‑4‑mini (GGUF 4‑bit) -----------------------------------------
@st.cache_resource
def get_llm():
    from huggingface_hub import hf_hub_download
    from ctransformers import AutoModelForCausalLM

    # 1) repo + filename corretos
    repo_id = "tensorblock/Phi-4-mini-instruct-GGUF"
    filename = "Phi-4-mini-instruct-Q4_K_M.gguf"  # note os hífens e maiúsculas

    # 2) faz o download (e cache) para a VM
    model_path = hf_hub_download(repo_id=repo_id, filename=filename, token=HF_TOKEN)

    # 3) carrega o GGUF em CPU
    llm = AutoModelForCausalLM.from_pretrained(
        model_path, model_type="phi3", context_length=4096, gpu_layers=0
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

    # como ctransformers já trata tokenizer e geração:
    resposta = llm(prompt, max_new_tokens=256, temperature=0.1)
    return resposta.strip()
