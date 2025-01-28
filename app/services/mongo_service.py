from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from typing import List, Dict, Optional
from app.models.conversation import Conversation, Message
from app.core.config import settings

class MongoService:
    def __init__(self):
        self.client = AsyncIOMotorClient(settings.mongodb_uri)
        self.db = self.client[settings.database_name]
        self.conversations = self.db[settings.collection_name]
        
    async def save_message(self, session_id: str, role: str, content: str) -> bool:
        """Sauvegarde un nouveau message dans une conversation"""
        message = Message(role=role, content=content)
        result = await self.conversations.update_one(
            {"session_id": session_id},
            {
                "$push": {"messages": message.model_dump()},
                "$set": {"updated_at": datetime.utcnow()},
                "$setOnInsert": {"created_at": datetime.utcnow()}
            },
            upsert=True
        )
        return result.modified_count > 0 or result.upserted_id is not None


    async def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Récupère l'historique d'une conversation"""
        conversation = await self.conversations.find_one({"session_id": session_id})
        if conversation:
            return conversation.get("messages", [])
        return []
    
    async def delete_conversation(self, session_id: str) -> bool:
        """Supprime une conversation"""
        result = await self.conversations.delete_one({"session_id": session_id})
        return result.deleted_count > 0
    
    # async def get_all_sessions(self) -> List[str]:
    #     """Récupère tous les IDs de session"""
    #     cursor = self.conversations.find({}, {"session_id": 1})
    #     sessions = await cursor.to_list(length=None)
    #     return [session["session_id"] for session in sessions]

    async def get_all_sessions(self) -> List[str]:
        cursor = self.conversations.find({}, {"session_id": 1})
        sessions = []
        async for session in cursor:
            sessions.append(session["session_id"])
        return sessions

    
    def get_collection(self, collection_name: str):
        return self.db[collection_name]