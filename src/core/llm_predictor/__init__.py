"""LLM Predictor Module.

This module provides a unified interface for interacting with different Large Language Model
(LLM) providers. It includes implementations for multiple providers (NVIDIA, OpenAI, Gemini)
and a factory pattern for easy instantiation.

The module exports:
    - LLMPredictor: Abstract base class defining the predictor interface
    - LLMFactory: Factory class for creating provider-specific predictors
    - NVIDIAPredictor: Concrete implementation for NVIDIA API
    - OpenAIPredictor: Concrete implementation for OpenAI API
    - GeminiPredictor: Concrete implementation for Google Gemini API

Example:
    >>> from src.core.llm_predictor import LLMFactory
    >>> predictor = LLMFactory.get_predictor("nvidia", "meta/llama-3-8b")
    >>> response = predictor.predict(
    ...     system_prompt="You are a helpful assistant",
    ...     user_prompt="Hello!",
    ...     temperature=0.7,
    ...     max_output_tokens=100
    ... )
"""

from src.core.llm_predictor.llm_predictor import LLMPredictor
from src.core.llm_predictor.llm_factory import LLMFactory
from src.core.llm_predictor.nvidia_predictor import NVIDIAPredictor

__all__ = [
    'LLMPredictor',
    'LLMFactory',
    'NVIDIAPredictor'
]
