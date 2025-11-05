from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class Settings(BaseSettings):
    GOOGLE_API_KEY: str

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    EMBEDDING_MODEL_NAME: str = "gemini-embedding-001"

    CHROMA_PERSIST_DIRECTORY: str = "chroma_db"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
