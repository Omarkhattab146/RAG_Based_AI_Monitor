from .providers.CopilotProvider import CopilotProvider
from .LLMenums import LLMProvider
from .providers import LocalEmbeddingProvider

# Take config and return the configured generation or embedding provider
class LLMPROVIDEFACTORY:
    """LLM provider factory for generation and embedding backends."""
    def __init__(self, config):
        self.config = config

    def create(self, provider: str):
        """Create provider instance for generation or embedding."""
        if provider == LLMProvider.COPILOT.value:
            return CopilotProvider()
        if provider == LLMProvider.SENTENCE_TRANSFORMER.value:
            return LocalEmbeddingProvider(
                default_input_max_chars=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_output_max_tokens=self.config.GENERATION_DEFAULT_MAX_TOKENS,
                temprature=self.config.GENERATION_DEFAULT_TEMPRETURE,
            )
        raise ValueError(f"Unknown LLM provider: {provider}. Supported: {[p.value for p in LLMProvider]}")