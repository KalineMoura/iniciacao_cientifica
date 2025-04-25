# src/main.py
from pathlib import Path
import os
from dotenv import load_dotenv
import streamlit as st
import torch   # precisamos verificar GPU

# ── env & timeout HF Hub ----------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HF_HUB_REQUEST_TIMEOUT"] = "60,300"

# 🟢  habilita back-end CPU do bitsandbytes caso não haja GPU
if not torch.cuda.is_available():
    os.environ["BNB_CUDALESS"] = "1"

# ── caminhos ----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "data" / "chunks_exemplos.md"

# ── embeddings + retriever --------------------------------------------------
@st.cache_resource
def get_retriever():
    from langchain_community.document_loaders import TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from huggingface_hub import snapshot_download

    repo_dir = snapshot_download("BAAI/bge-small-en-v1.5", token=HF_TOKEN)

    loader = TextLoader(DATA_FILE)
    docs = loader.load()

    embeddings = HuggingFaceEmbeddings(
        model_name=repo_dir,
        model_kwargs={"device": "cpu"},
    )
    vect = FAISS.from_documents(docs, embeddings)
    return vect.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 4})

# ── LLM : Phi-3-mini-4k-instruct em 8-bit -----------------------------------
@st.cache_resource
def get_llm():
    from huggingface_hub import snapshot_download
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )

    repo_dir = snapshot_download(
        "microsoft/Phi-3-mini-4k-instruct", token=HF_TOKEN
    )

    # quantização 8-bit (mais estável em CPU)
    bnb_cfg = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,   # padrão seguro
    )

    tokenizer = AutoTokenizer.from_pretrained(repo_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo_dir,
        trust_remote_code=True,
        quantization_config=bnb_cfg,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        low_cpu_mem_usage=True,
    )
    return tokenizer, model

# ── função pública ----------------------------------------------------------
def gerar_resposta(pergunta: str) -> str:
    """Gera resposta do assistente financeiro usando RAG + LLM."""
    retriever = get_retriever()
    tokenizer, model = get_llm()

    docs = retriever.get_relevant_documents(pergunta)
    contexto = "\n\n".join(d.page_content[:1000] for d in docs) or "N/D"

    prompt = f"""
Você é um assistente financeiro. Com base no seguinte contexto,
responda de forma clara e objetiva. Caso não saiba, responda que não sabe.

Contexto:
{contexto}

Pergunta:
{pergunta}
"""

    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(
        inputs,
        max_new_tokens=256,
        temperature=0.1
    )[0]
    full = tokenizer.decode(output, skip_special_tokens=True)
    return full[len(prompt):].strip()
