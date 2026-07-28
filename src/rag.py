from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from ingest import load_documents


chunks = load_documents()


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


db = FAISS.from_documents(
    chunks,
    embeddings
)


retriever = db.as_retriever(
    search_kwargs={"k":3}
)


endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens=512,
    temperature=0
)


llm = ChatHuggingFace(
    llm=endpoint
)


prompt = ChatPromptTemplate.from_template(
"""
Responde únicamente usando el contexto.

Contexto:
{context}

Pregunta:
{question}
"""
)


def format_docs(docs):
    return "\n\n".join(
        doc.page_content 
        for doc in docs
    )


qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)