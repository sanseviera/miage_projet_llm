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
from services.rag_service import RAGService

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

                # Ajout du service RAG
        self.rag_service = RAGService()
        
        # Mise à jour du prompt pour inclure le contexte RAG
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Vous êtes un assistant utile et concis. "
                      "Utilisez le contexte fourni pour répondre aux questions."),
            ("system", "Contexte : {context}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
    
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
    async def generate_response(self, 
                              message: str, 
                              context: Optional[List[Dict[str, str]]] = None,
                              session_id: Optional[str] = None,
                              use_rag: bool = False) -> str:
        """Méthode mise à jour pour supporter le RAG"""
        rag_context = ""
        if use_rag and self.rag_service.vector_store:
            relevant_docs = await self.rag_service.similarity_search(message)
            rag_context = "\n\n".join(relevant_docs)
        
        if session_id:
            response = await self.chain_with_history.ainvoke(
                {
                    "question": message,
                    "context": rag_context
                },
                config={"configurable": {"session_id": session_id}}
            )
            return response.content
        else:
            messages = [
                SystemMessage(content="Vous êtes un assistant utile et concis.")
            ]
            
            if rag_context:
                messages.append(SystemMessage(
                    content=f"Contexte : {rag_context}"
                ))
            
            if context:
                for msg in context:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
            
            messages.append(HumanMessage(content=message))
            response = await self.llm.agenerate([messages])
            response_text = response.generations[0][0].text
            return response_text

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