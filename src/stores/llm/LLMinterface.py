from abc import ABC, abstractmethod


class LLMInterface(ABC):
    @abstractmethod
    def set_generation_model(self, model_id: str):
        """Set which LLM model to use for text generation"""
        pass
    
    @abstractmethod
    def set_embedding_model(self, model_id: str, embedding_size: int):
        """Set which model to use for creating embeddings"""
        pass
    
    @abstractmethod
    def generate_text(self, prompt: str, chat_history: list = [],
                     max_output_tokens: int = None,
                     temperature: float = None):
        """Generate text based on prompt and conversation history"""
        pass
    
    @abstractmethod
    def embed_text(self, text: str, doc_type: str = None):
        """Convert text into a numerical vector (embedding)"""
        pass
    
    @abstractmethod
    def construct_prompt(self, prompt: str, role: str):
        """Format prompt for the LLM API"""
        pass

