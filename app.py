import streamlit as st
from src.rag import qa_chain


st.title("🤖 Agente RAG AIESEC")


pregunta = st.text_input(
    "Escribe tu pregunta:"
)


if pregunta:

    respuesta = qa_chain.invoke(
        pregunta
    )

    st.write("Respuesta:")
    st.write(respuesta.content)