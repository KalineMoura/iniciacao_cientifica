# src/main.py
from pathlib import Path
import os, time
from dotenv import load_dotenv

import streamlit as st  # ← para cache_resource
from langchain_core.messages import AIMessage, HumanMessage

# ── env --------------------------------------------------------------------
load_dotenv()  # carrega secrets do Streamlit tb.
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ── caminhos ----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent  # .../src
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
        model_name="BAAI/bge-small-en-v1.5",  # leve, multilíngue
        model_kwargs={"device": "cpu"},
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 4},
    )


# ── LLM (Phi‑3‑mini 4‑bit) --------------------------------------------------
@st.cache_resource
def get_llm():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch, accelerate

    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True)
    model_id = "microsoft/phi-3-mini-4k-instruct"  # ~3.8 B, cabe em 4‑bit

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    return tok, model


# ── função pública ----------------------------------------------------------
def gerar_resposta(pergunta: str) -> str:
    """Gera resposta do assistente financeiro."""
    retriever = get_retriever()
    tokenizer, llm = get_llm()

    docs_relev = retriever.get_relevant_documents(pergunta)
    contexto = "\n\n".join(doc.page_content[:1000] for doc in docs_relev) or "N/D"

    prompt = f"""
    Você é um assistente financeiro. Com base no seguinte contexto,
    responda de forma clara e objetiva. Caso não saiba, responda
    que não sabe.

    Contexto:
    {contexto}

    Pergunta:
    {pergunta}
    """

    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(llm.device)
    output = llm.generate(ids, max_new_tokens=256, temperature=0.1, do_sample=False)[0]
    full = tokenizer.decode(output, skip_special_tokens=True)

    return full[len(prompt) :].strip()
