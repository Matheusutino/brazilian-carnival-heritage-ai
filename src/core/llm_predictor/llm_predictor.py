"""Base abstract class for LLM predictors.

This module defines the interface that all LLM predictor implementations must follow.
It provides an abstract base class that enforces a consistent API across different
language model providers.
"""

from abc import ABC, abstractmethod


class LLMPredictor(ABC):
    """Abstract base class defining the interface for LLM predictors.

    This class establishes the contract that all concrete LLM predictor implementations
    must adhere to. Each implementation should provide a predict method that handles
    communication with its respective language model API.
    """

    @abstractmethod
    def predict(self, system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
        """Generate a response from the language model.

        Args:
            system_prompt: System-level instructions that set the behavior and context
                for the language model.
            user_prompt: The user's input or query to be processed by the model.
            temperature: Controls randomness in the output. Higher values (e.g., 1.0)
                make output more random, lower values (e.g., 0.1) make it more deterministic.
            max_output_tokens: Maximum number of tokens to generate in the response.

        Returns:
            The model's response as a string.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        pass