from fastapi import FastAPI
from api.routes.qa_router import router as qa_router # Modified import
import uvicorn

app = FastAPI(title="LegalAI Backend by Langchain")
app.include_router(qa_router,prefix="/api")

@app.get("/")
def read_root():
    return {"message":"Welcome to LegalAI API by Langchain"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)