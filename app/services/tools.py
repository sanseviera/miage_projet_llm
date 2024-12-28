from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, ZeroShotAgent
import requests
from typing import Dict, List
import json

class AssistantTools:
    def __init__(self, llm):
        self.llm = llm
        self.tools = self._initialize_tools()
        self.agent_executor = self._initialize_agent()
    
    def _initialize_tools(self) -> List[Tool]:
        tools = [
            Tool(
                name="Calculatrice",
                func=self._calculate,
                description="Utile pour effectuer des calculs mathématiques"
            ),
            Tool(
                name="RechercheWeb",
                func=self._search_web,
                description="Recherche des informations sur le web"
            ),
            Tool(
                name="Traducteur",
                func=self._translate,
                description="Traduit du texte entre différentes langues"
            )
        ]
        return tools
    
    def _initialize_agent(self) -> AgentExecutor:
        prompt = ZeroShotAgent.create_prompt(
            self.tools,
            prefix="Vous êtes un assistant intelligent. Utilisez ces outils pour aider l'utilisateur:",
            suffix="Question: {input}\nPensée: Réfléchissons étape par étape:\n{agent_scratchpad}"
        )
        
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain, tools=self.tools)
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True
        )
    
    def _calculate(self, expression: str) -> str:
        """Effectue un calcul mathématique"""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Erreur de calcul: {str(e)}"
    
    def _search_web(self, query: str) -> str:
        """Simule une recherche web"""
        # Implémentez votre logique de recherche ici
        return f"Résultats pour: {query}"
    
    def _translate(self, text: str) -> str:
        """Simule une traduction"""
        # Implémentez votre logique de traduction ici
        return f"Traduction de: {text}"
    
    async def process_request(self, query: str) -> str:
        """Traite une requête utilisateur avec les outils disponibles"""
        try:
            response = await self.agent_executor.arun(query)
            return response
        except Exception as e:
            return f"Erreur lors du traitement: {str(e)}"