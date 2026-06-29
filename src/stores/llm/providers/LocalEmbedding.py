from ..LLMinterface import LLMInterface
import logging # for making logs messages with all actions done

from sentence_transformers import SentenceTransformer
import time

class LocalEmbeddingProvider(LLMInterface):
    def __init__(self, # api_key: str,
                 default_input_max_chars: int = 1000,
                 default_output_max_tokens: int = 150,
                 temprature:float=0.1):
        
        # self.api_key = api_key
        self.default_input_max_chars = default_input_max_chars
        self.default_output_max_tokens = default_output_max_tokens
        self.temprature = temprature

        self.generation_model_id = None
        
        self.embedding_model_id = None
        self.embedding_size = None

        # self.client = cohere.Client(api_key=self.api_key)
        self.client = None
        
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

        self.client = SentenceTransformer(model_id)



    def process_text(self, text:str):
        return text[:self.default_input_max_chars] # take first x chars from user prompt


    def generate_text(self, prompt: str, chat_history: list = [], max_output_tokens: int = None,
                       temperature: float = None):
        pass

    
    def embed_text(self, text: str, doc_type: str = None):
        if not self.client:
            self.logger.error("Model not loaded. Call set_embedding_model first.")
            return None


        try:
            if doc_type == "query":
                text = f"Represent this sentence for searching relevant passages: {text}"
            processed_text = self.process_text(text)
            embedding = self.client.encode(processed_text)
            return embedding.tolist()  # ✅ حول لـ list
        except Exception as e:
            self.logger.error(f"Embedding failed: {e}")
            return None



    def embed_texts(self, texts: list, doc_type: str = None, batch_size: int = 32):
        """✅ تحويل قائمة نصوص لـ embeddings"""
        if not self.client:
            self.logger.error("Model not loaded. Call set_embedding_model first.")
            return None

        try:
            if doc_type == "query":
                texts = [f"Represent this sentence for searching relevant passages: {text}" for text in texts]
            processed_texts = [self.process_text(t) for t in texts]
            
            # ✅ Sentence Transformers بيعمل batching تلقائي
            print(f"🔄 Embedding {len(processed_texts)} texts...")
            embeddings = self.client.encode(
                processed_texts, 
                batch_size=batch_size,
                show_progress_bar=True
            )
            
            print(f"✅ Generated {len(embeddings)} embeddings")
            return embeddings.tolist()  # ✅ حول لـ list
            
        except Exception as e:
            self.logger.error(f"Batch embedding failed: {e}")
            return None

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "text": self.process_text(prompt)
        }