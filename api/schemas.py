# Request and response structures for FastAPI
from pydantic import BaseModel

class DocumentRequest(BaseModel):
    content : str
    
class QueryRequest(BaseModel):
    question : str

class QueryResponse(BaseModel):
    answer : str
    