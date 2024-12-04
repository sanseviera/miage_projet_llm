from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import os

class LLMService:
    def __init__(self):
        # Récupération de la clé API OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY n'est pas définie dans les variables d'environnement")
            
        print(f"API Key trouvée : {api_key[:5]}...") # Affiche les 5 premiers caractères pour vérifier
        
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            api_key=api_key
        )
    async def generate_response(self, message: str, context: list = None) -> str:
        messages = []
        
        # Ajout du contexte système initial
        messages.append(SystemMessage(content="Vous êtes un assistant utile et concis."))
        
        # Ajout de l'historique des messages si présent
        if context:
            for msg in context:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        
        # Ajout du message actuel
        messages.append(HumanMessage(content=message))
        
        # Génération de la réponse
        response = await self.llm.agenerate([messages])
        return response.generations[0][0].text