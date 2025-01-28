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
import logging 
import re
from pymongo.collection import Collection
import uuid

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
            ("system", "Vous êtes un assistant spécialisé uniquement en ressources humaines. Vous êtes un assistant linguistique IA avec une expertise en ressources humaines (RH)."
            "Votre objectif principal est de fournir des réponses détaillées et précises liées au recrutement, à la gestion des talents, à l'engagement des employés, aux politiques RH, à la rémunération et aux avantages sociaux, à la conformité et au développement organisationnel,"
            "aux politiques RH et aux sujets connexes. "
            "Assurez-vous que vos réponses sont pertinentes pour les professionnels des RH ou les demandeurs d'emploi."
            "Évitez les interprétations qui ne sont pas liées aux RH, sauf demande explicite."
            "Pour les questions hors de ce domaine, répondez : 'Je suis désolé, je ne suis spécialisé que dans les ressources humaines.'"),
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
    

    async def get_all_conversation_ids(self) -> List[str]:
        collection = self.mongo_service.get_collection("conversations")
        cursor = collection.find({}, {"session_id": 1})
        sessions = await cursor.to_list(length=None)  
        return [doc["session_id"] for doc in sessions]
    
    async def create_new_session(self) -> str:
        collection = self.mongo_service.get_collection("conversations")
        
        cursor = collection.find({"session_id": {"$regex": "^session_\\d+$"}}, {"session_id": 1})
        sessions = await cursor.to_list(length=None)
        
        session_numbers = []
        for doc in sessions:
            match = re.match(r"session_(\d+)", doc["session_id"])
            if match:
                session_numbers.append(int(match.group(1)))
        
        if session_numbers:
            new_number = max(session_numbers) + 1
        else:
            new_number = 1
        
        session_id = f"session_{new_number}"
        new_session = {
            "session_id": session_id,
            # "created_at": datetime.utcnow(),
            # "last_activity": datetime.utcnow(),
            # "message_count": 0,
            # "tags": [],
            # "summary": "",
        }
        await collection.insert_one(new_session)
        return session_id