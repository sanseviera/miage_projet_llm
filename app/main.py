from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router import router as api_router
from services.llm_service import LLMService
import uvicorn
from services.rag_service import RAGService
from fastapi.responses import JSONResponse
import json

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, Heroku!"}


load_dotenv()

app = FastAPI(
    title="Agent conversationnel",
    description="API pour un agent conversationnel donné lors du TP1",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instance du service LLM
llm_service = LLMService()
rag_service = RAGService()

# Inclure les routes
app.include_router(api_router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)