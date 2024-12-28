from pydantic_settings import BaseSettings
import logging
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

class Settings(BaseSettings):
    mongodb_uri: str
    database_name: str = "chatbot"
    collection_name: str = "conversations"

    class Config:
        """Classe de configuration pour MongoDB."""
        MONGO_URI = os.getenv("MONGO_URI", "")  # Valeur par défaut vide si non définie
        MONGO_DB = os.getenv("MONGO_DB", "default_db")  # Nom de base par défaut

settings = Settings()