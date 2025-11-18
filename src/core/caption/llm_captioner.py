"""Image caption generation module using multimodal Large Language Models.

This module provides rich caption generation capabilities using multimodal LLMs
with support for structured outputs including descriptions, keywords, and cultural analysis.
"""
import json
import re
import random
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import yaml
from jsonschema import validate, ValidationError, Draft7Validator

from src.core.llm_predictor.llm_factory import LLMFactory
from src.core.utils.image_encoding import encode_image_for_llm


class LLMCaptioner:
    """Image captioner using multimodal LLM with retry mechanism.

    This class generates rich, structured captions for images using multimodal LLMs.
    It includes detailed descriptions, keywords, visual elements, temporal context,
    cultural analysis, and technical quality assessment.

    Attributes:
        config (dict): Loaded YAML configuration containing system and user prompts.
        provider (str): LLM provider name.
        model_name (str): Model identifier.
        temperature (float): Initial temperature for LLM generation.
        max_tokens (int): Maximum tokens for LLM response.
        max_retries (int): Maximum number of retry attempts.
        retry_temp_min (float): Minimum temperature for retry attempts.
        retry_temp_max (float): Maximum temperature for retry attempts.
        llm: LLM predictor instance from LLMFactory.
        lock (Lock): Thread lock for thread-safe operations.
    """

    def __init__(self, config_path: str, provider: str, model_name: str,
                 temperature: float, max_tokens: int, max_retries: int = 3,
                 retry_temp_min: float = 0.0, retry_temp_max: float = 1.0,
                 prompt_key: str = 'caption',
                 max_image_dim: int | None = None):
        """Initialize image captioner.

        Args:
            config_path (str): Path to YAML configuration file with prompts.
            provider (str): LLM provider name (e.g., 'nvidia', 'openai').
            model_name (str): Model identifier.
            temperature (float): Temperature for first attempt.
            max_tokens (int): Maximum output tokens.
            max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.
            retry_temp_min (float, optional): Minimum temperature for retry attempts.
                Defaults to 0.0.
            retry_temp_max (float, optional): Maximum temperature for retry attempts.
                Defaults to 1.0.
            prompt_key (str, optional): Key in config YAML for prompts. Defaults to 'caption'.

        Raises:
            FileNotFoundError: If config_path does not exist.
            yaml.YAMLError: If the configuration file is not valid YAML.
            KeyError: If prompt_key does not exist in configuration.
        """
        config_data = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
        self.config = config_data[prompt_key]
        self.prompt_key = prompt_key
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_temp_min = retry_temp_min
        self.retry_temp_max = retry_temp_max
        self.llm = LLMFactory.get_predictor(provider, model_name)
        self.lock = Lock()
        self.max_image_dim = max_image_dim

        # Load JSON schema for validation if available
        self.output_schema = self.config.get('output_schema', None)
        if self.output_schema:
            # Create validator for better error messages
            self.validator = Draft7Validator(self.output_schema)

    def _encode_image(self, image_path: Path) -> str:
        """Encode image as base64, resizing when max_image_dim is configured."""
        max_dim = self.max_image_dim
        encoded, _ = encode_image_for_llm(
            image_path,
            max_dim=max_dim,
            max_b64_length=45000 if max_dim else None
        )
        return encoded

    def _build_prompt(self, image_b64: str) -> tuple:
        """Build system and user prompts with image.

        Constructs the system and user prompts by replacing the image placeholder
        with the base64-encoded image.

        Args:
            image_b64 (str): Base64-encoded image string.

        Returns:
            tuple: A tuple containing (system_prompt, user_prompt) strings.
        """
        system = self.config['system_prompt']
        user = self.config['user_prompt']

        # Embed base64 image as data URI in HTML img tag
        user = user.replace('{image_tag}', f'<img src="data:image/png;base64,{image_b64}" />')

        return system, user

    def _extract_json(self, response: str):
        """Extract and validate JSON from LLM response.

        Attempts to extract and parse JSON from the LLM response. First tries to
        find JSON within markdown code blocks (```json...```), then falls back to
        parsing the entire response as JSON. If a schema is configured, validates
        the JSON against the schema.

        Args:
            response (str): Raw LLM response string.

        Returns:
            tuple: (parsed_json or None, error_message or None)
                - If successful: (dict, None)
                - If failed: (None, error_message_string)
        """
        parsed_json = None

        # Try to extract JSON from markdown code block first
        match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            try:
                parsed_json = json.loads(match.group(1))
            except json.JSONDecodeError as e:
                return None, f"Invalid JSON in code block: {str(e)}"

        # Try parsing the entire response as JSON if code block failed
        if parsed_json is None:
            try:
                parsed_json = json.loads(response)
            except json.JSONDecodeError as e:
                return None, f"Failed to parse JSON: {str(e)}"

        # Validate against schema if available
        if self.output_schema and parsed_json:
            try:
                validate(instance=parsed_json, schema=self.output_schema)
                return parsed_json, None
            except ValidationError as e:
                # Build detailed error message
                error_path = " -> ".join(str(p) for p in e.path) if e.path else "root"
                error_msg = f"Schema validation failed at '{error_path}': {e.message}"

                # Add missing required fields info if available
                if e.validator == 'required':
                    missing = set(e.validator_value) - set(e.instance.keys())
                    error_msg += f". Missing required fields: {', '.join(missing)}"

                return None, error_msg

        # No schema or validation passed
        return parsed_json, None

    def generate_caption(self, image_path: Path, metadata: dict = None):
        """Generate caption for a single image with retry mechanism.

        Attempts to generate a structured caption for an image using the LLM.
        If the first attempt fails to extract valid JSON, retries up to max_retries
        times with randomized temperature.

        Args:
            image_path (Path): Path to the image file.
            metadata (dict, optional): Optional metadata dictionary to include in results.
                Defaults to None.

        Returns:
            dict: Dictionary containing caption results with the following keys:
                - image_path (str): String representation of the image path.
                - image_name (str): Name of the image file.
                - timestamp (str): ISO format timestamp of captioning.
                - status (str): 'success' or 'error'.
                - metadata (dict): Provided metadata or empty dict.
                - attempts (list): List of attempt information dictionaries.
                - caption (dict): Structured caption results (only if successful).
                - temperature (float): Temperature used for successful attempt (or None).
                - raw_response (str): Raw LLM response (only if successful).
                - error (str): Error message (only if failed).
        """
        result = {
            'image_path': str(image_path),
            'image_name': image_path.name,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending',
            'metadata': metadata or {},
            'attempts': []
        }

        img_b64 = None

        # Retry loop with temperature randomization
        for attempt in range(1, self.max_retries + 1):
            # First attempt uses configured temperature, retries use random values
            temp = self.temperature if attempt == 1 else random.uniform(self.retry_temp_min, self.retry_temp_max)

            try:
                # Encode image to base64 only once (cache for retries)
                if img_b64 is None:
                    img_b64 = self._encode_image(image_path)

                # Build prompts with image
                system, user = self._build_prompt(img_b64)

                # Send request to LLM
                response = self.llm.predict(system, user, temp, self.max_tokens)

                # Try to extract and validate JSON from response
                caption_data, extraction_error = self._extract_json(response)

                # Record attempt metadata
                attempt_info = {
                    'attempt': attempt,
                    'temperature': temp,
                    'success': caption_data is not None,
                    'error': extraction_error
                }
                result['attempts'].append(attempt_info)

                # If successful, store results and break
                if caption_data:
                    result['caption'] = caption_data
                    result['status'] = 'success'
                    result['temperature'] = temp
                    result['raw_response'] = response
                    break

            except Exception as e:
                # Record failed attempt with error details
                attempt_info = {
                    'attempt': attempt,
                    'temperature': temp,
                    'success': False,
                    'error': str(e)
                }
                result['attempts'].append(attempt_info)

        # Mark as error if all attempts failed
        if result['status'] == 'pending':
            result['status'] = 'error'
            result['error'] = f'Failed after {self.max_retries} attempts'
            result['temperature'] = None

        return result

    def generate_batch(self, images: list, metadatas: list = None, max_workers: int = 4, verbose: bool = True):
        """Generate captions for multiple images in parallel.

        Uses a thread pool to generate captions for multiple images concurrently.
        Progress is printed if verbose is True.

        Args:
            images (list): List of image paths (Path objects or strings).
            metadatas (list, optional): Optional list of metadata dictionaries, one per image.
                Must have same length as images if provided. Defaults to None.
            max_workers (int, optional): Number of parallel worker threads. Defaults to 4.
            verbose (bool, optional): If True, prints progress information during processing.
                Defaults to True.

        Returns:
            list: List of caption result dictionaries, one per image. See generate_caption()
                method for result structure.
        """
        metadatas = metadatas or [None] * len(images)
        results = []
        processed = 0

        if verbose:
            print(f"\nGenerating captions for {len(images)} images ({max_workers} workers)...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.generate_caption, img, meta): (img, meta)
                for img, meta in zip(images, metadatas)
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                with self.lock:
                    processed += 1
                    if verbose:
                        icon = "✓" if result['status'] == 'success' else "✗"
                        print(f"[{processed}/{len(images)}] {icon} {result['image_name']}")

        if verbose:
            success = sum(1 for r in results if r['status'] == 'success')
            print(f"\n✓ Complete: {success}/{len(results)} successful")

        return results
