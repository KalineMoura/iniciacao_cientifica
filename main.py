from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from langchain_core.messages import AIMessage, HumanMessage
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st
import time
from dotenv import load_dotenv
import os

# Carrega variáveis do .env
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Carrega markdown já chunkado
loader = TextLoader("/content/chunks_exemplos.md")
docs = loader.load()
splits = docs

# Indexação com embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"}
)
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 3, 'fetch_k': 4})

# Carrega modelo local phi-4-mini
phi_model_id = "microsoft/phi-4-mini-instruct"
phi_tokenizer = AutoTokenizer.from_pretrained(phi_model_id)
phi_model = AutoModelForCausalLM.from_pretrained(
    phi_model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

# Inicializa histórico no session_state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Função que responde perguntas
def responder(pergunta):
    try:
        if not st.session_state.chat_history:
            st.session_state.chat_history.append(
                AIMessage(content="Olá! Me envie sua dúvida.")
            )

        docs_relevantes = retriever.get_relevant_documents(pergunta)
        contexto = "\n\n".join([doc.page_content[:1000] for doc in docs_relevantes])

        prompt = f"""
        Você é um assistente financeiro. Com base no seguinte contexto, responda à pergunta do usuário de forma clara e objetiva. Se não souber, diga que não sabe.

        Contexto:
        {contexto}

        Pergunta:
        {pergunta}
        """

        input_ids = phi_tokenizer(prompt, return_tensors="pt").input_ids.to(phi_model.device)
        output = phi_model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False
        )[0]

        full_text = phi_tokenizer.decode(output, skip_special_tokens=True)
        resposta = full_text[len(prompt):].strip()

        st.session_state.chat_history.append(HumanMessage(content=pergunta))
        st.session_state.chat_history.append(AIMessage(content=resposta))

        placeholder = st.empty()
        buffer = ""
        for char in resposta:
            buffer += char
            placeholder.markdown(buffer)
            time.sleep(0.015)

    except Exception as e:
        st.error(f"Erro: {str(e)}")
