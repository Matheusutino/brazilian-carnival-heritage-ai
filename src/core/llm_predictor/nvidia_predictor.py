"""NVIDIA API predictor implementation.

This module provides a concrete implementation of the LLMPredictor interface for
NVIDIA's language models. It leverages NVIDIA's OpenAI-compatible API endpoint,
allowing the use of the OpenAI SDK with NVIDIA's infrastructure.
"""

import os
from openai import OpenAI
from src.core.llm_predictor.llm_predictor import LLMPredictor
from dotenv import load_dotenv, find_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv(find_dotenv())


class NVIDIAPredictor(LLMPredictor):
    """Predictor implementation using NVIDIA's API (OpenAI-compatible).

    This class provides integration with NVIDIA's language models through their
    OpenAI-compatible API endpoint. It uses the OpenAI SDK configured with NVIDIA's
    base URL and includes automatic retry logic for handling transient API failures.

    Attributes:
        api_key: NVIDIA API key loaded from environment variables.
        base_url: NVIDIA's OpenAI-compatible API endpoint URL.
        client: OpenAI client configured to use NVIDIA's API endpoint.
        model: Name of the NVIDIA model to use for predictions.
    """

    def __init__(self, model: str):
        """Initialize the NVIDIA predictor with the specified model.

        Args:
            model: The name of the NVIDIA model to use (e.g., specific NVIDIA
                model identifiers compatible with their API).

        Note:
            Requires NVIDIA_API_KEY to be set in environment variables.
        """
        self.api_key = os.getenv("NVIDIA_API_KEY")
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = model

    @retry(
        stop=stop_after_attempt(3),                     # Attempt up to 3 times
        wait=wait_exponential(multiplier=1, min=2, max=5),  # Exponential backoff between attempts
        retry=retry_if_exception_type(Exception),       # Can be customized for specific error types
        reraise=True                                     # Re-raise the error if all attempts fail
    )
    def predict(self, system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
        """Generate a response using NVIDIA's language model API.

        This method sends a request to NVIDIA's OpenAI-compatible API with the provided
        prompts and configuration parameters. It includes automatic retry logic with
        exponential backoff to handle transient API failures.

        Args:
            system_prompt: System-level instructions that set the behavior and context
                for the language model.
            user_prompt: The user's input or query to be processed by the model.
            temperature: Controls randomness in the output. Higher values (e.g., 1.0)
                make output more random, lower values (e.g., 0.1) make it more deterministic.
            max_output_tokens: Maximum number of tokens to generate in the response.

        Returns:
            The model's response text with leading/trailing whitespace removed.

        Raises:
            Exception: If all retry attempts fail, the last exception is re-raised.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_output_tokens
        )
        return response.choices[0].message.content.strip()
