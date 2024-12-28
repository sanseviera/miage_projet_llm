from datetime import datetime
from fastapi import APIRouter, HTTPException
from models.chat import ChatRequestTP1, ChatRequestTP2, ChatRequestWithContext, ChatResponse, SummaryResponse, SummaryRequest, ChatRequestAdv, MemoryTagRequest, MemoryClearRequest, MetadataResponse, ToolRequest
from services.llm_service import LLMService
from typing import Dict, List

router = APIRouter()
llm_service = LLMService()

@router.post("/chat/simple", response_model=ChatResponse)
async def chat_simple(request: ChatRequestTP2) -> ChatResponse:
    """Endpoint simple du TP1"""
    try:
        response = await llm_service.generate_response(
             message=request.message,
             session_id=request.session_id
            )
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/with-context", response_model=ChatResponse)
async def chat_with_context(request: ChatRequestWithContext) -> ChatResponse:
    """Endpoint avec contexte du TP1"""
    try:
        response = await llm_service.generate_response(
            message=request.message,
            session_id=request.session_id,
            context=request.context
        )
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequestTP2) -> ChatResponse:
    """Nouvel endpoint du TP2 avec gestion de session"""
    try:
        response = await llm_service.generate_response(
            message=request.message,
            session_id=request.session_id
        )
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""
Service de résumé de texte multi-niveaux
"""
@router.post("/summarize", response_model=SummaryResponse)
async def summarize_text(request: SummaryRequest):
    try:
        summary = await llm_service.generate_summary(request.text)
        return SummaryResponse(**summary)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")

@router.get("/history/{session_id}")
async def get_history(session_id: str) -> List[Dict[str, str]]:
    """Récupération de l'historique d'une conversation"""
    try:
        raw_history = await llm_service.get_conversation_history(session_id)
        # raw_history est une liste de dict { "role":..., "content":..., "timestamp": ... }

        # Convertir chaque 'timestamp' en isoformat (ou un autre format de chaîne)
        history_as_str = []
        for item in raw_history:
            # On copie tout l'item
            converted_item = dict(item)
            if "timestamp" in converted_item and isinstance(converted_item["timestamp"], datetime):
                converted_item["timestamp"] = converted_item["timestamp"].isoformat()
            history_as_str.append(converted_item)

        return history_as_str  # On renvoie la liste "typée" comme List[Dict[str, str]]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/advanced", response_model=ChatResponse)
async def chat_advanced(request: ChatRequestAdv) -> ChatResponse:
    """
    Endpoint qui permet de choisir la mémoire avancée ou non
    """
    try:
        response = await llm_service.generate_response(
            message=request.message,
            session_id=request.session_id,
            use_advanced=request.use_advanced_memory
        )
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/chat/advanced/add-tag")
async def add_tag_to_advanced_memory(request: MemoryTagRequest):
    """
    Ajoute un 'tag' à la session AdvancedMemory.
    """
    try:
        # 1) Récupérer la session advanced
        chat_history_obj = llm_service._get_session_history_advanced(request.session_id)
        # 2) Appeler la méthode add_tag
        chat_history_obj.add_tag(request.tag)
        # 3) Retourner un message de succès
        return {"msg": f"Tag '{request.tag}' ajouté à la session '{request.session_id}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/advanced/clear")
async def clear_advanced_memory(request: MemoryClearRequest):
    """
    Vide l'historique de la session advanced
    """
    try:
        chat_history_obj = llm_service._get_session_history_advanced(request.session_id)
        chat_history_obj.clear()
        return {"msg": f"Session '{request.session_id}' effacée."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/advanced/metadata/{session_id}", response_model=MetadataResponse)
async def get_advanced_metadata(session_id: str):
    """
    Retourne la metadata de la session advanced (created_at, last_activity, nb de messages, tags, etc.)
    """
    try:
        chat_history_obj = llm_service._get_session_history_advanced(session_id)
        # Récupérer l'objet metadata
        meta = chat_history_obj.metadata
        
        return MetadataResponse(
            session_id=session_id,
            created_at=meta.created_at.isoformat(),
            last_activity=meta.last_activity.isoformat(),
            message_count=meta.message_count,
            tags=meta.tags,
            summary=meta.summary,
            is_active=chat_history_obj.is_active()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/assistant/tool", response_model=ChatResponse)
async def use_tool(request: ToolRequest):
    try:
        response = await llm_service.process_with_tools(request.message)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
