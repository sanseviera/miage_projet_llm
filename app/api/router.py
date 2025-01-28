from fastapi import APIRouter
from api.endpoints import chat

router = APIRouter()

router.include_router(
    chat.router, 
    prefix="/chat", 
    tags=["chat"]
)