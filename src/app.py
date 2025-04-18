# src/app.py
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from main import gerar_resposta  # ← importa do src.main

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

# ---- exibe histórico -------------------------------------------------------
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
        resposta = gerar_resposta(pergunta)

        # animação de “digitando…”
        buf = ""
        for ch in resposta:
            buf += ch
            placeholder.markdown(buf)
            st.sleep(0.015)

        st.session_state.chat_history.append(AIMessage(content=resposta))
