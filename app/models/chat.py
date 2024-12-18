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
