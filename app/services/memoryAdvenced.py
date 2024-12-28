from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

from datetime import datetime, timedelta
from typing import Dict, Optional, List
import json

class ConversationMetadata:
    def __init__(self):
        self.created_at: datetime = datetime.now()
        self.last_activity: datetime = datetime.now()
        self.message_count: int = 0
        self.tags: List[str] = []
        self.summary: Optional[str] = None

class EnhancedMemoryHistory(BaseChatMessageHistory):
    def __init__(self, 
                 max_messages: int = 50,
                 session_timeout: timedelta = timedelta(hours=1)):
        self.messages: List[BaseMessage] = []
        self.max_messages = max_messages
        self.session_timeout = session_timeout
        self.metadata = ConversationMetadata()

    def __iter__(self):
        return iter(self.messages)
        
    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Ajoute des messages avec gestion de la limite"""
        self.metadata.last_activity = datetime.now()
        self.metadata.message_count += len(messages)
        
        # Ajout des messages avec respect de la limite
        self.messages.extend(messages)
        if len(self.messages) > self.max_messages:
            # Garder les messages les plus récents
            self.messages = self.messages[-self.max_messages:]
            
        # Mise à jour du résumé si nécessaire
        if self.metadata.message_count % 10 == 0:  # Tous les 10 messages
            self._update_summary()
    
    def add_tag(self, tag: str) -> None:
        """Ajoute un tag à la conversation"""
        if tag not in self.metadata.tags:
            self.metadata.tags.append(tag)
    
    def is_active(self) -> bool:
        """Vérifie si la session est encore active"""
        return (datetime.now() - self.metadata.last_activity) < self.session_timeout
    
    async def _update_summary(self) -> None:
        """Met à jour le résumé de la conversation"""
        if not self.messages:
            return
            
        # Utilisation du LLM pour générer un résumé
        messages_text = "\n".join([msg.content for msg in self.messages[-10:]])
        summary_prompt = f"Résumez brièvement cette conversation :\n{messages_text}"
        
        try:
            # Vous devrez adapter cette partie selon votre configuration LLM
            self.metadata.summary = await self.llm.agenerate([summary_prompt])
        except Exception as e:
            print(f"Erreur lors de la mise à jour du résumé : {e}")

    def clear(self) -> None:
        """
        Implémentation requise par la classe parente BaseChatMessageHistory.
        Permet de vider l'historique si nécessaire.
        """
        self.messages = []
        self.metadata = ConversationMetadata()