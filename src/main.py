# src/main.py
from pathlib import Path
import os
from dotenv import load_dotenv
import streamlit as st
import time
from requests.exceptions import ReadTimeout

# ── env & timeout HF Hub ---------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# Aumenta timeout p/ as requisições do HF Hub (connect, read)
os.environ["HF_HUB_REQUEST_TIMEOUT"] = "60,300"

# ── caminhos ----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "data" / "chunks_exemplos.md"


# ── embeddings + retriever --------------------------------------------------
@st.cache_resource
def get_llm():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    model_id = "microsoft/phi-3-mini-instruct"  # modelo compatível
    # quantização 4-bit leve
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True)

    # autoriza custom code e carrega tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # carrega o modelo em 4-bit via bitsandbytes
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    return tokenizer, model


# ── LLM  : Phi-4-mini (GGUF 4-bit) -----------------------------------------
@st.cache_resource
def get_llm():
    from huggingface_hub import hf_hub_download
    from ctransformers import AutoModelForCausalLM

    # ↓ repos correto e nome exato do arquivo
    repo_id = "microsoft/Phi-3-mini-4k-instruct-gguf"
    filename = "Phi-3-mini-4k-instruct-q4.gguf"

    model_path = hf_hub_download(repo_id=repo_id, filename=filename, token=HF_TOKEN)

    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        model_type="phi-3-mini",  # para phi-3-mini-4k-instruct GGUF
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
