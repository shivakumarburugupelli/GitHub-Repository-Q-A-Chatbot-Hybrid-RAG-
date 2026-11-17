from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from rag import load_repository, answer_question
from github_loader import GitHubLoaderError
app = FastAPI(
    title="GitHub Repository Q&A API"
)
class Repository(BaseModel):

    github_url: str
class Question(BaseModel):

    question: str

@app.get("/")
def home():    

    return {
        "message": "GitHub RAG API is running"
    }

@app.post("/load")
def load_repo(data: Repository):

    try:
        result = load_repository(
            data.github_url
        )
    except GitHubLoaderError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=str(error)
        )

    return {
        "message": result
    }

@app.post("/ask")

def ask(data: Question):

    result = answer_question(
        data.question
    )

    return {
        "question": data.question,
        "answer": result["answer"],
        "sources": result["sources"]
    }