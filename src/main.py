# src/main.py
from pathlib import Path
import os
from dotenv import load_dotenv
import streamlit as st

# ── env & timeout HF Hub ---------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# aumenta timeout para downloads do HF Hub
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

    # pré-baixa e cacheia o modelo de embeddings
    repo_dir = snapshot_download(repo_id="BAAI/bge-small-en-v1.5", token=HF_TOKEN)

    # carrega os documentos chunked
    loader = TextLoader(DATA_FILE)
    docs = loader.load()

    # inicializa embeddings a partir do cache local
    embeddings = HuggingFaceEmbeddings(
        model_name=repo_dir,
        model_kwargs={"device": "cpu"},
    )
    vect = FAISS.from_documents(docs, embeddings)
    return vect.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 4})


# ── LLM  : Phi-3-mini-instruct em 4-bit -------------------------------------
@st.cache_resource
def get_llm():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    model_id = "microsoft/phi-3-mini-instruct"
    # quantização leve em 4-bit
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True)

    # carrega tokenizer e modelo com bitsandbytes
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    return tokenizer, model


# ── função pública ----------------------------------------------------------
def gerar_resposta(pergunta: str) -> str:
    retriever = get_retriever()
    tokenizer, llm = get_llm()

    # busca contexto relevante
    docs = retriever.get_relevant_documents(pergunta)
    contexto = "\n\n".join(d.page_content[:1000] for d in docs) or "N/D"

    # monta prompt
    prompt = f"""Você é um assistente financeiro. Com base no seguinte contexto,
responda de forma clara e objetiva. Caso não saiba, responda que não sabe.

Contexto:
{contexto}

Pergunta:
{pergunta}
"""

    # gera resposta
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(llm.device)
    output = llm.generate(inputs, max_new_tokens=256, temperature=0.1)[0]
    full = tokenizer.decode(output, skip_special_tokens=True)

    # retorna só o trecho gerado além do prompt
    return full[len(prompt) :].strip()
