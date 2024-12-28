from typing import Any, Dict
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

class SummaryService:
    def __init__(self, llm):
        self.llm = llm
        
        # Chaîne pour le résumé complet
        self.full_summary_prompt = PromptTemplate(
            template="""Résumez le texte suivant en conservant les points importants :
            
            {text}
            
            Résumé détaillé :""",
            input_variables=["text"]
        )
        self.full_summary_chain = LLMChain(
            llm=self.llm,
            prompt=self.full_summary_prompt,
            output_key="full_summary"
        )
        
        # Chaîne pour les points clés
        self.bullet_points_prompt = PromptTemplate(
            template="""Extrayez les 3-5 points clés de ce résumé :
            
            {full_summary}
            
            Points clés :""",
            input_variables=["full_summary"]
        )
        self.bullet_points_chain = LLMChain(
            llm=self.llm,
            prompt=self.bullet_points_prompt,
            output_key="bullet_points"
        )
        
        # Chaîne pour la phrase de synthèse
        self.one_liner_prompt = PromptTemplate(
            template="""Résumez ces points clés en une seule phrase percutante :
            
            {bullet_points}
            
            Phrase de synthèse :""",
            input_variables=["bullet_points"]
        )
        self.one_liner_chain = LLMChain(
            llm=self.llm,
            prompt=self.one_liner_prompt,
            output_key="one_liner"
        )
        
        # Chaîne séquentielle complète
        self.summary_chain = SequentialChain(
            chains=[
                self.full_summary_chain,
                self.bullet_points_chain,
                self.one_liner_chain
            ],
            input_variables=["text"],
            output_variables=["full_summary", "bullet_points", "one_liner"]
        )
    
    async def generate_summary(self, text: str) -> Dict[str, Any]:
        try:
            # Étape 1 : Résumé complet
            full_summary_result = await self.full_summary_chain.arun({"text": text})
            
            # Étape 2 : Points clés
            bullet_points_result = await self.bullet_points_chain.arun({"full_summary": full_summary_result})
            
            # Étape 3 : Phrase de synthèse
            one_liner_result = await self.one_liner_chain.arun({"bullet_points": bullet_points_result})
            
            return {
                "full_summary": full_summary_result,
                "bullet_points": bullet_points_result.split("\n"),
                "one_liner": one_liner_result
            }
        except Exception as e:
            raise ValueError(f"Erreur lors de la génération du résumé : {str(e)}")