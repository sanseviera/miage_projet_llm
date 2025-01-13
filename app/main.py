from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router import router as api_router
from services.llm_service import LLMService
import uvicorn
from services.rag_service import RAGService
from fastapi.responses import JSONResponse
import json

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

@app.get("/chat/documents")
async def get_documents():
    try:
        documents = rag_service.get_all_documents()
        formatted_response = {
            "documents": documents
        }
        return JSONResponse(content=json.dumps(formatted_response, indent=4))
    except ValueError as e:
        return JSONResponse(content=json.dumps({"error": str(e)}, indent=4))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)