# src/main.py
from pathlib import Path
import os, time
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.messages import AIMessage, HumanMessage

import torch

# ── env --------------------------------------------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ── caminhos ----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent  # .../src
DATA_FILE = BASE_DIR.parent / "data" / "chunks_exemplos.md"


# ── embeddings + retriever (cache simples) ----------------------------------
def _build_retriever():
    loader = TextLoader(DATA_FILE)
    docs = loader.load()

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 4},
    )


RETRIEVER = _build_retriever()

# ── modelo LLM (Phi‑4‑mini) --------------------------------------------------
phi_id = "microsoft/phi-4-mini-instruct"
PHI_TOKENIZER = AutoTokenizer.from_pretrained(phi_id)
PHI_MODEL = AutoModelForCausalLM.from_pretrained(
    phi_id, device_map="auto", torch_dtype=torch.float16
)


# ── função pública ----------------------------------------------------------
def gerar_resposta(pergunta: str) -> str:
    """Retorna uma string com a resposta do assistente."""
    docs_relev = RETRIEVER.get_relevant_documents(pergunta)
    contexto = "\n\n".join(doc.page_content[:1000] for doc in docs_relev)

    prompt = f"""
    Você é um assistente financeiro. Com base no seguinte contexto,
    responda de forma clara e objetiva. Caso não saiba, responda
    que não sabe.

    Contexto:
    {contexto}

    Pergunta:
    {pergunta}
    """

    ids = PHI_TOKENIZER(prompt, return_tensors="pt").input_ids.to(PHI_MODEL.device)
    output = PHI_MODEL.generate(
        ids, max_new_tokens=512, temperature=0.1, do_sample=False
    )[0]

    full = PHI_TOKENIZER.decode(output, skip_special_tokens=True)
    return full[len(prompt) :].strip()
