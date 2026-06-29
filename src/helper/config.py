# using pydantic to manage configuration settings from environment variables
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings # Basemodel here is to make scheme for all project

# Define a settings class to manage configuration using Pydantic by taking values from environment variables
class Settings(BaseSettings):
    # Configuration for Pydantic settings
    APP_NAME: str
    APP_VERSION: str 
    #
    FILE_ALOWED_EXTENSTIONS: list
    FILE_MAX_SIZE: int

    FILE_DEFAULT_CHUNK_SIZE: int
    FILE_CHUNK_OVERLAP: int = 100
    FILE_UPLOAD_BATCH_SIZE: int = 64
    
    MONGO_URI: str
    MONGODB_DATABASE: str


    GENERATION_BACKEND : str 
    EMBEDDING_BACKEND : str

    # GitHub Models / Copilot
    GITHUB_TOKEN: Optional[str] = None
    GITHUB_MODELS_MODEL: str = "gpt-4o"
    COPILOT_NUM_PARALLEL: int = 4
    COPILOT_CONTEXT_LENGTH: int = 4096
    COPILOT_REQUEST_GAP_SECONDS: float = 20.0
    COPILOT_ENDPOINT: str = "https://models.github.ai/inference"
    COPILOT_MODEL: str = "openai/gpt-5"

    GENERATION_MODEL_ID : str = None
    EMBEDDING_MODEL_ID : str = None
    EMBEDDING_MODEL_SIZE : int = None

    INPUT_DEFAULT_MAX_CHARACTERS : int = None
    GENERATION_DEFAULT_MAX_TOKENS : int = None
    GENERATION_DEFAULT_TEMPRETURE : float = None
    GENERATION_CONTEXT_MAX_CHARS : int = 40000
    GENERATION_CONTEXT_MAX_CHUNKS : int = 12
    QA_RETRIEVAL_PAGE_SIZE: int = 500
    QA_TASK_REQUEST_GAP_SECONDS: float = 20.0
    QA_SORT_QUESTIONS_BY_TYPE: bool = True
    RETRIEVAL_TOP_K: int = 20
    RETRIEVAL_SCORE_THRESHOLD: float = 0.6

    VECTOR_DB_BACKEND: str 
    VECTOR_DB_PATH: str 
    VECTOR_DB_DISTANCE_METHOD: str = None
    VECTOR_DB_COLLECTION_PREFIX: str = "project_"

    PRIMARY_LANG: str = "en"
    DEFAULT_LANG: str = "en"

    # Main backend base URL for ingest webhooks (no trailing slash required).
    # Local: https://localhost:7080 | Deployed: https://ailern.runasp.net
    BACKEND_WEBHOOK_BASE_URL: Optional[str] = None
    # Set False for local HTTPS when the backend uses a dev self-signed cert (e.g. ASP.NET).
    # Keep True in production.
    BACKEND_WEBHOOK_VERIFY_SSL: bool = True
    # Transient webhook failures: attempts and exponential backoff (initial * 2**n, capped).
    BACKEND_WEBHOOK_MAX_ATTEMPTS: int = 3
    BACKEND_WEBHOOK_RETRY_INITIAL_SECONDS: float = 1.0
    BACKEND_WEBHOOK_RETRY_MAX_SECONDS: float = 30.0

    class Config:
        # Specify the .env file to load environment variables from
        env_file = ".env"
        extra = "ignore"

@lru_cache()
def get_settings():
    return Settings()

