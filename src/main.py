from rag import qa_chain

pregunta = input("Pregunta: ")

respuesta = qa_chain.invoke(pregunta)

print(respuesta.content)