from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import os
from typing import List, Dict
from services.mongo_service import MongoService

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from services.memory import InMemoryHistory
import os
from typing import List, Dict, Optional

class LLMService:

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY n'est pas définie")
        
        # Configuration commune
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            api_key=api_key
        )
        
        # Configuration pour le TP2
        self.conversation_store = {}
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Vous êtes un assistant utile et concis."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        
        self.chain = self.prompt | self.llm
        
        # Configuration du gestionnaire d'historique
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self._get_session_history,
            input_messages_key="question",
            history_messages_key="history"
        )
        self.mongo_service = MongoService()
    
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Récupère ou crée l'historique pour une session donnée"""
        if session_id not in self.conversation_store:
            self.conversation_store[session_id] = InMemoryHistory()
        return self.conversation_store[session_id]
    
    async def generate_response(self,message: str,session_id: str) -> str:
        """Génère une réponse et sauvegarde dans MongoDB"""
        # Récupération de l'historique depuis MongoDB
        history = await self.mongo_service.get_conversation_history(session_id)
        # Conversion de l'historique en messages LangChain
        messages = [SystemMessage(content="Vous êtes un assistant utile et concis.")]
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        # Ajout du nouveau message
        messages.append(HumanMessage(content=message))
        # Génération de la réponse
        response = await self.llm.agenerate([messages])
        response_text = response.generations[0][0].text
        # Sauvegarde des messages dans MongoDB
        await self.mongo_service.save_message(session_id, "user", message)
        await self.mongo_service.save_message(session_id, "assistant", response_text)
        return response_text
    
async def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
    """Récupère l'historique depuis MongoDB"""
    return await self.mongo_service.get_conversation_history(session_id)