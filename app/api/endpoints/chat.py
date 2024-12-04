from fastapi import APIRouter, HTTPException
from models.chat import ChatRequest, ChatRequestWithContext, ChatResponse
from services.llm_service import LLMService


router = APIRouter()

llm_service = LLMService()

@router.post("/chat/simple", response_model=ChatResponse)
async def chat_simple(request: ChatRequest) -> ChatResponse:
    try:
        response = await llm_service.generate_response(request.message)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/with-context", response_model=ChatResponse)
async def chat_with_context(request: ChatRequestWithContext) -> ChatResponse:
    try:
        response = await llm_service.generate_response(
            message=request.message,
            context=request.context
        )
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))