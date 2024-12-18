from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    mongodb_uri: str
    database_name: str = "chatbot"
    collection_name: str = "conversations"

class Config:
    env_file = ".env"

settings = Settings()