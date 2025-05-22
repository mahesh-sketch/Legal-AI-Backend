from fastapi import APIRouter,HTTPException
from api.schemas import DocumentRequest,QueryResponse,QueryRequest
from langchain_modules.embedder import embed_text_to_faiss
from langchain_modules.langgraph_flow import qa_flow_graph
from core.logger import logger

router = APIRouter()

graph = qa_flow_graph()

@router.post("/embed")
def embed_doc(request:DocumentRequest):
    try:
        response = embed_text_to_faiss(request.content)
        logger.info(response["message"])
        return response
    except Exception as e:
        logger.info("Embedding failed")
        raise HTTPException(status_code=500,detail=str(e))


@router.post("/ask",response_model=QueryResponse)
def ask_question(query:QueryRequest):
    try:
        result = graph.invoke({"question": query.question})
        return result
    except ValueError as ve:
        if "Vector store is not initialized" in str(ve):
            logger.error(f"ValueError during QA: {str(ve)}")
            raise HTTPException(status_code=400, detail="Vector store is not initialized. Please embed a document first via the /embed endpoint.")
        else:
            logger.exception(f"Unhandled ValueError during QA: {str(ve)}")
            raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        logger.exception("Error during QA") # Log other exceptions
        raise HTTPException(status_code=500, detail=str(e))