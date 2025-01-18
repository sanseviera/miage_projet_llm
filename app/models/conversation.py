from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Conversation(BaseModel):
    session_id: str
    messages: List[Message] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class SummaryRequest(BaseModel):
    text: str
    max_length: int

class ChatRequestAdv(BaseModel):
    message: str
    session_id: str
    use_advanced_memory: bool = False

class IndexDocumentsRequest(BaseModel):
    texts: List[str]
    clear_existing: bool = False