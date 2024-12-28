from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import os
from typing import Any, List, Dict
from services.mongo_service import MongoService

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from services.memory import InMemoryHistory
from services.memoryAdvenced import EnhancedMemoryHistory
from services.chains import SummaryService
from services.tools import AssistantTools
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

        # Configuration de MongoDB
        self.mongo_service = MongoService()

        # Configuration du service de résumé pour le TP2 exo 1 (voir services/chains.py)
        self.summary_service = SummaryService(self.llm)

        # Configuration pour l'Assistant avec Outils
        self.tools = AssistantTools(self.llm)
    
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Récupère ou crée l'historique pour une session donnée"""
        if session_id not in self.conversation_store:
            self.conversation_store[session_id] = InMemoryHistory()
        return self.conversation_store[session_id]
    
    def _get_session_history_advanced(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.conversation_store:
            self.conversation_store[session_id] = EnhancedMemoryHistory()
        return self.conversation_store[session_id]
    
    """async def generate_response(self,message: str,session_id: str) -> str:
        #Génère une réponse et sauvegarde dans MongoDB
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
        return response_text"""
    async def generate_response(
        self,
        message: str,
        session_id: str,
        context: Optional[List[Dict[str, str]]] = None,
        use_advanced: bool = False
    ) -> str:
        """Génère une réponse et sauvegarde dans MongoDB, avec contexte optionnel."""
        
        # 1) Récupération de l'historique depuis MongoDB
        #history = await self.mongo_service.get_conversation_history(session_id)
        # Vérifier si use_advanced == True
        if use_advanced:
            chat_history_obj = self._get_session_history_advanced(session_id)
        else:
            chat_history_obj = self._get_session_history(session_id)

        # 2) Conversion de l'historique DB en messages LangChain
        messages = [SystemMessage(content="Vous êtes un assistant utile et concis.")]
        for msg in chat_history_obj:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        # 3) Incorporer le 'context' supplémentaire s'il existe
        if context:
            for cmsg in context:
                if cmsg["role"] == "user":
                    messages.append(HumanMessage(content=cmsg["content"]))
                elif cmsg["role"] == "assistant":
                    messages.append(AIMessage(content=cmsg["content"]))

        # 4) Ajouter le nouveau message utilisateur
        messages.append(HumanMessage(content=message))

        # 5) Génération de la réponse via LLM
        response = await self.llm.agenerate([messages])
        response_text = response.generations[0][0].text

        # 6) Sauvegarde dans MongoDB
        await self.mongo_service.save_message(session_id, "user", message)
        await self.mongo_service.save_message(session_id, "assistant", response_text)

        return response_text
    
    async def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Récupère l'historique depuis MongoDB"""
        return await self.mongo_service.get_conversation_history(session_id)
    
    """ Exo 2 : Ajout de la méthode pour le TP2 """
    # Ajout de la méthode pour générer un résumé
    async def generate_summary(self, text: str) -> Dict[str, Any]:
        return await self.summary_service.generate_summary(text)
    
    def cleanup_inactive_sessions(self):
        """Nettoie les sessions inactives"""
        current_time = datetime.now()
        for session_id, history in list(self.conversation_store.items()):
            if not history.is_active():
                del self.conversation_store[session_id]

    # Ajout de la méthode pour l'appel aux outils de l'assistant
    async def process_with_tools(self, query: str) -> str:
            return await self.tools.process_request(query)