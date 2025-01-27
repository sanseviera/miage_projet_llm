from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import os
from typing import Any, List, Dict
from app.services.mongo_service import MongoService

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from app.services.memory import InMemoryHistory
from app.services.memoryAdvenced import EnhancedMemoryHistory
from app.services.chains import SummaryService
from app.services.tools import AssistantTools
import os
from typing import List, Dict, Optional
from app.services.rag_service import RAGService
import logging 
import re
from pymongo.collection import Collection

class LLMService:

    def __init__(self):
        self.mongo_service = MongoService("DATABASE_NAME")

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
        self.rag_service = RAGService()
        
        # Configuration pour le TP2
        self.conversation_store = {}
        # self.prompt = ChatPromptTemplate.from_messages([
        #     ("system", "Vous êtes un assistant utile et concis."),
        #     MessagesPlaceholder(variable_name="history"),
        #     ("human", "{question}")
        # ])
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Vous êtes un assistant utile et concis. "
                      "Utilisez le contexte fourni pour répondre aux questions."),
            ("system", "Contexte : {context}"),
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
    
    async def generate_response(self, message: str, session_id: str, context: Optional[List[Dict[str, str]]] = None, use_rag: bool = False) -> str:
        # Retrieve conversation history
        history = await self.mongo_service.get_conversation_history(session_id)

        # Initialize RAG context
        rag_context = ""
        if use_rag:
            # Fetch relevant documents for RAG
            relevant_docs = await self.rag_service.similarity_search(message)
            if relevant_docs:
                # rag_context = "\n\n".join([doc.page_content for doc in relevant_docs])
                rag_context = "\n\n".join([f"- {doc.page_content}" for doc in relevant_docs])
                logging.info(f"RAG Context generated: {rag_context}")

        # Include context in messages for the LLM
        messages = [SystemMessage(content="You are a helpful assistant.")]
        if rag_context:
            messages.append(SystemMessage(content=f"Context: {rag_context}"))

        # Add history to messages
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        # Add user query
        messages.append(HumanMessage(content=message))

        # Generate response with the chain
        response = await self.chain_with_history.ainvoke(
            {"question": message, "context": rag_context},
            config={"configurable": {"session_id": session_id}}
        )
        response_text = response.content

        # Save the messages to MongoDB
        await self.mongo_service.save_message(session_id, "user", message)
        await self.mongo_service.save_message(session_id, "assistant", response_text)

        return response_text

    
    async def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Récupère l'historique depuis MongoDB"""
        return await self.mongo_service.get_conversation_history(session_id)
    """ Exo 2 : Ajout de la méthode pour le TP2 """
    # Ajout de la méthode pour générer un résumé
    async def generate_summary(self, text: str, max_length: int) -> Dict[str, Any]:
        try:
            print("text", text)
            logging.info(f"Original text: {text}")
            sanitized_text = text.replace(" ", " ").replace("\n", " ")
            print(sanitized_text)
            logging.info(f"Sanitized text: {sanitized_text}")
            result = await self.summary_service.generate_summary(sanitized_text, max_length)
            logging.info(f"Summary result: {result}")
            return result
        except Exception as e:
            logging.error(f"Error during summary generation: {str(e)}")
            raise ValueError(f"Erreur lors de la génération du résumé : {str(e)}")

    
    def cleanup_inactive_sessions(self):
        """Nettoie les sessions inactives"""
        for session_id, history in list(self.conversation_store.items()):
            if not history.is_active():
                del self.conversation_store[session_id]

    # Ajout de la méthode pour l'appel aux outils de l'assistant
    async def process_with_tools(self, query: str) -> str:
            return await self.tools.process_request(query)
    
    # async def get_all_conversation_ids(self) -> List[str]:
    #     try:
    #         collection: Collection = self.mongo_service.get_collection("conversations")
    #         session_ids = collection.distinct("session_id")
    #         return session_ids
    #     except Exception as e:
    #         logging.error(f"Error retrieving conversation IDs: {str(e)}")
    #         raise ValueError(f"Error retrieving conversation IDs: {str(e)}")

    async def get_all_conversation_ids(self) -> List[str]:
        collection = self.mongo_service.get_collection("conversations")
        cursor = collection.find({}, {"session_id": 1})
        sessions = await cursor.to_list(length=None)  # Correctly handle cursor
        return [doc["session_id"] for doc in sessions]
