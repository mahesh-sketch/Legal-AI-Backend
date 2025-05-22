from langchain.chains import RetrievalQA 
from langchain_google_genai import ChatGoogleGenerativeAI
from . import embedder 
from core.config import settings
from core.logger import logger


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=settings.GOOGLE_API_KEY)

def answer_question(question:str):
    logger.info(f"qa_chain.py: Accessing embedder.vector_store, id: {id(embedder.vector_store)}")
    if embedder.vector_store is None:
        logger.error("Vector store is None in qa_chain.answer_question. Please embed a document first.")
        raise ValueError("Vector store is not initialized. Please embed a document first.")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=embedder.vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False
    )
    result = qa_chain.run(question)
    return {"answer": result}
