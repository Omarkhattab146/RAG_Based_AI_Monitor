from enum import Enum

class LLMProvider(Enum):
    # Generation backend: Copilot (GitHub Models)
    COPILOT = "COPILOT"
    # Embedding backend: local sentence transformer utility
    SENTENCE_TRANSFORMER = "SentenceTransformer"


class Document(Enum):
    DOCUMENT = "document"
    QUERY = "query" 