from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional
from pydantic import BaseModel, Field

# models/chat.py
"""
Modèles Pydantic pour la validation des données
Inclut les modèles du TP1 et les nouveaux modèles pour le TP2
"""

# ---- Modèles du TP1 ----
class ChatRequestTP1(BaseModel):
    """Requête de base pour une conversation sans contexte"""
    message: str

class ChatResponse(BaseModel):
    """Réponse standard du chatbot"""
    response: str

class ChatRequestWithContext(BaseModel):
    """Requête avec contexte de conversation du TP1"""
    message: str
    session_id: str
    context: Optional[List[Dict[str, str]]] = []

# ---- Nouveaux modèles pour le TP2 ----

class ChatRequestTP2(BaseModel):
    """Requête de base pour une conversation sans contexte"""
    message: str
    session_id: str  # Ajouté pour supporter les deux versions

class ChatMessage(BaseModel):
    """Structure d'un message individuel dans l'historique"""
    role: str  # "user" ou "assistant"
    content: str

class ChatHistory(BaseModel):
    """Collection de messages formant une conversation"""
    messages: List[ChatMessage]

# ---- Nouveaux modèles pour l'exo 1 ----
class SummaryRequest(BaseModel):
    text: str
    max_length: Optional[int] = 1000

class SummaryResponse(BaseModel):
    full_summary: str
    bullet_points: List[str]
    one_liner: str

class HistoryMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime  # On accepte un datetime

class ChatRequestAdv(BaseModel):
    message: str
    session_id: str
    use_advanced_memory: bool = False

class MemoryTagRequest(BaseModel):
    session_id: str
    tag: str

class MemoryClearRequest(BaseModel):
    session_id: str

class MetadataResponse(BaseModel):
    session_id: str
    created_at: str
    last_activity: str
    message_count: int
    tags: List[str]
    summary: Optional[str] = None
    is_active: bool

# Pour l'assitant avec outils
class ToolRequest(BaseModel):
    message: str
    session_id: str
    tool: str