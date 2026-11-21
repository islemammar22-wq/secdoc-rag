"""FastAPI service. Run: uvicorn app:app --reload"""
from fastapi import FastAPI
from pydantic import BaseModel, Field

import rag

app = FastAPI(title="SecDoc RAG", version="0.1")


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    k: int = Field(default=5, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    return rag.answer(req.question, req.k)