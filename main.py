from fastapi import FastAPI, UploadFile, File
from app.llm_chain import answer_question
from app.pdf_loader import load_pdf
from app.vector_store import create_vectorstore
from app.models import QueryRequest, QueryResponse

app = FastAPI()

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    return await answer_question(request.question)

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()
    docs = load_pdf(content)
    create_vectorstore(docs)
    return {"status": "PDF processed successfully"}
