from typing import Dict, List, Optional
from pydantic import BaseModel, Field

# Version simple
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# Version avec contexte
class ChatRequestWithContext(BaseModel):
    message: str
    context: Optional[List[Dict[str, str]]] = [] 