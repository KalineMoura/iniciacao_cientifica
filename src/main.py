# src/main.py
from pathlib import Path
import os
from dotenv import load_dotenv
import streamlit as st

# ── env & timeout HF Hub ---------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# Aumenta timeout para downloads do HF Hub (conexão e leitura)
os.environ["HF_HUB_REQUEST_TIMEOUT"] = "60,300"

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

    # Pré-baixa e cacheia o modelo de embeddings BGE-small
    repo_dir = snapshot_download(repo_id="BAAI/bge-small-en-v1.5", token=HF_TOKEN)

    # Carrega documentos chunked
    loader = TextLoader(DATA_FILE)
    docs = loader.load()

    # Inicializa embeddings a partir do cache local
    embeddings = HuggingFaceEmbeddings(
        model_name=repo_dir,
        model_kwargs={"device": "cpu"},
    )
    vect = FAISS.from_documents(docs, embeddings)
    return vect.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 4})


# ── LLM : Phi-3-mini-4k-instruct em 4-bit via bitsandbytes ---------------
@st.cache_resource
def get_llm():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    # Pré-baixa e cacheia o repositório do LLM
    repo_dir = snapshot_download(
        repo_id="microsoft/Phi-3-mini-4k-instruct", token=HF_TOKEN
    )

    # Configura quantização 4-bit
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True)

    # Carrega tokenizer e modelo a partir da pasta local
    tokenizer = AutoTokenizer.from_pretrained(repo_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo_dir,
        trust_remote_code=True,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    return tokenizer, model


# ── função pública ----------------------------------------------------------
def gerar_resposta(pergunta: str) -> str:
    """Gera resposta do assistente financeiro usando RAG + LLM."""
    # Recupera o retriever e o LLM (cada um em cache)
    retriever = get_retriever()
    tokenizer, model = get_llm()

    # Obtém contexto relevante
    docs = retriever.get_relevant_documents(pergunta)
    contexto = "\n\n".join(d.page_content[:1000] for d in docs) or "N/D"

    # Prepara prompt
    prompt = f"""
Você é um assistente financeiro. Com base no seguinte contexto,
responda de forma clara e objetiva. Caso não saiba, responda que não sabe.

Contexto:
{contexto}

Pergunta:
{pergunta}
"""

    # Gera saída
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(inputs, max_new_tokens=256, temperature=0.1)[0]
    full = tokenizer.decode(output, skip_special_tokens=True)

    # Retorna apenas a parte da resposta
    return full[len(prompt) :].strip()
