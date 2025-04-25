########################################################
import torch, bitsandbytes as bnb, accelerate
print("torch:", torch.__version__,
      "| bnb:", bnb.__version__,
      "| accelerate:", accelerate.__version__,
      "| CUDA?", torch.cuda.is_available())
########################################################
# src/app.py
import streamlit as st
import time
from langchain_core.messages import AIMessage, HumanMessage
from main import gerar_resposta  # chama a função lazy‑cacheada

st.set_page_config(page_title="Assistente Financeiro", page_icon="💰")
st.title("💰 Assistente Financeiro")

# ---- estado da conversa ----------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá! Como posso ajudar nas suas finanças hoje?")
    ]

if st.button("🗑️ Limpar conversa"):
    st.session_state.chat_history = st.session_state.chat_history[:1]
    st.experimental_rerun()

# ---- exibe histórico existente --------------------------------------------
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ---- input do usuário ------------------------------------------------------
pergunta = st.chat_input("Digite sua pergunta…")

if pergunta:
    st.session_state.chat_history.append(HumanMessage(content=pergunta))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Gerando resposta..."):
            try:
                resposta = gerar_resposta(pergunta)
            except Exception as e:
                resposta = f"Desculpe, ocorreu um erro: {e}"

        # animação de “digitando…”
        buf = ""
        for ch in resposta:
            buf += ch
            placeholder.markdown(buf)
            time.sleep(0.012)

        st.session_state.chat_history.append(AIMessage(content=resposta))
