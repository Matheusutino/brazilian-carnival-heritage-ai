"""Factory pattern implementation for creating LLM predictors.

This module provides a factory class that dynamically instantiates the appropriate
LLM predictor based on the provider configuration. It centralizes predictor creation
logic and simplifies switching between different language model providers.
"""

from src.core.llm_predictor.llm_predictor import LLMPredictor
from src.core.llm_predictor.nvidia_predictor import NVIDIAPredictor


class LLMFactory:
    """Factory class for creating LLM predictor instances.

    This class implements the factory design pattern to dynamically create predictor
    instances based on the specified provider. It abstracts the instantiation logic
    and provides a unified interface for creating different types of predictors.
    """

    @staticmethod
    def get_predictor(llm_provider: str, model_name: str) -> LLMPredictor:
        """Create and return an LLM predictor for the specified provider.

        This static method instantiates the appropriate predictor implementation
        based on the provider name. Currently supports NVIDIA API.

        Args:
            llm_provider: The name of the LLM provider to use. Currently only
                "nvidia" is supported (case-sensitive).
            model_name: The specific model name to use with the selected provider
                (e.g., "meta/llama-4-maverick-17b-128e-instruct").

        Returns:
            An instance of the appropriate LLMPredictor subclass configured with
            the specified model.

        Raises:
            ValueError: If the specified provider is not supported.

        Examples:
            >>> predictor = LLMFactory.get_predictor("nvidia", "meta/llama-4-maverick-17b-128e-instruct")
        """
        if llm_provider == "nvidia":
            return NVIDIAPredictor(model_name)
        else:
            raise ValueError(f"Provider '{llm_provider}' is not supported.")