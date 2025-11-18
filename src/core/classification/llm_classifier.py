"""Image classification module using multimodal Large Language Models.

This module provides image classification capabilities using multimodal LLMs through
a unified interface. It supports multiple LLM providers and includes a robust retry
mechanism with temperature randomization for improved reliability.
"""
import json
import re
import random
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
import yaml
from jsonschema import validate, ValidationError, Draft7Validator

from src.core.llm_predictor.llm_factory import LLMFactory
from src.core.utils.image_encoding import encode_image_for_llm


class ImageClassifier:
    """Image classifier using multimodal LLM with retry mechanism.

    This class provides image classification by encoding images to base64 and sending
    them to multimodal LLMs along with category prompts. It includes a retry mechanism
    that randomizes temperature on failed attempts to improve success rate.

    Attributes:
        config (dict): Loaded YAML configuration containing system and user prompts.
        categories (list): List of category names for classification.
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

    def __init__(self, config_path: str, categories: list, provider: str, model_name: str,
                 temperature: float, max_tokens: int, max_retries: int = 3,
                 retry_temp_min: float = 0.0, retry_temp_max: float = 1.0,
                 prompt_key: str = 'classification/zero_shot_basic',
                 max_image_dim: int | None = None):
        """Initialize image classifier.

        Args:
            config_path (str): Path to YAML configuration file with prompts.
            categories (list): List of category names for classification.
            provider (str): LLM provider name (e.g., 'nvidia', 'openai').
            model_name (str): Model identifier.
            temperature (float): Temperature for first attempt.
            max_tokens (int): Maximum output tokens.
            max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.
            retry_temp_min (float, optional): Minimum temperature for retry attempts.
                Defaults to 0.0.
            retry_temp_max (float, optional): Maximum temperature for retry attempts.
                Defaults to 1.0.
            prompt_key (str, optional): Key path in config YAML for prompts (e.g., 'classification/zero_shot_basic').
                Defaults to 'classification/zero_shot_basic'.

        Raises:
            FileNotFoundError: If config_path does not exist.
            yaml.YAMLError: If the configuration file is not valid YAML.
            KeyError: If prompt_key does not exist in configuration.
        """
        config_data = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))

        # Navigate nested keys (e.g., 'classification/zero_shot_basic')
        keys = prompt_key.split('/')
        self.config = config_data
        for key in keys:
            self.config = self.config[key]

        self.prompt_key = prompt_key
        self.categories = categories
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
        """Encode image as base64, resizing when max_image_dim is configured.

        Loads an image from disk and encodes it as a base64 string suitable for
        sending to multimodal LLMs. If max_image_dim is configured, the image is
        resized proportionally to fit within the specified maximum dimension while
        preserving aspect ratio.

        Args:
            image_path (Path): Path to the image file to encode.

        Returns:
            str: Base64-encoded image string (without data URI prefix).

        Raises:
            FileNotFoundError: If the image file does not exist.
            PIL.UnidentifiedImageError: If the file is not a valid image.
        """
        max_dim = self.max_image_dim
        encoded, _ = encode_image_for_llm(
            image_path,
            max_dim=max_dim,
            max_b64_length=45000 if max_dim else None
        )
        return encoded

    def _build_prompt(self, image_b64: str) -> tuple:
        """Build system and user prompts with categories.

        Constructs the system and user prompts by replacing placeholders in the
        configuration with actual category list and base64-encoded image.

        Args:
            image_b64 (str): Base64-encoded image string.

        Returns:
            tuple: A tuple containing (system_prompt, user_prompt) strings.
        """
        system = self.config['system_prompt']

        # Format category list as bullet points with expected values (0 or 1)
        cats_list = '\n  '.join([f"- {cat}: 0 or 1" for cat in self.categories])

        # Replace placeholders in user prompt template
        user = self.config['user_prompt']
        user = user.replace('{categories_list}', cats_list)
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

    def classify(self, image_path: Path, metadata: dict = None):
        """Classify a single image with retry mechanism.

        Attempts to classify an image using the LLM. If the first attempt fails to
        extract valid JSON, retries up to max_retries times with randomized temperature.

        Args:
            image_path (Path): Path to the image file.
            metadata (dict, optional): Optional metadata dictionary to include in results.
                Defaults to None.

        Returns:
            dict: Dictionary containing classification results with the following keys:
                - image_path (str): String representation of the image path.
                - image_name (str): Name of the image file.
                - timestamp (str): ISO format timestamp of classification.
                - status (str): 'success' or 'error'.
                - metadata (dict): Provided metadata or empty dict.
                - attempts (list): List of attempt information dictionaries.
                - classification (dict): Classification results (only if successful).
                - temperature (float): Temperature used for successful attempt (or None).
                - raw_response (str): Raw LLM response (only if successful).
                - error (str): Error message (only if failed).
        """
        # Accept string paths for compatibility with loaders that return raw strings
        if isinstance(image_path, str):
            image_path = Path(image_path)

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

                # Build prompts with image and categories
                system, user = self._build_prompt(img_b64)

                # Send request to LLM
                response = self.llm.predict(system, user, temp, self.max_tokens)

                # Try to extract and validate JSON from response
                classification, extraction_error = self._extract_json(response)

                # Record attempt metadata
                attempt_info = {
                    'attempt': attempt,
                    'temperature': temp,
                    'success': classification is not None,
                    'error': extraction_error
                }
                result['attempts'].append(attempt_info)

                # If successful, store results and break
                if classification:
                    result['classification'] = classification
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

    def classify_batch(self, images: list, metadatas: list = None, max_workers: int = 4, verbose: bool = True):
        """Classify multiple images in parallel.

        Uses a thread pool to classify multiple images concurrently. Progress is
        printed if verbose is True.

        Args:
            images (list): List of image paths (Path objects or strings).
            metadatas (list, optional): Optional list of metadata dictionaries, one per image.
                Must have same length as images if provided. Defaults to None.
            max_workers (int, optional): Number of parallel worker threads. Defaults to 4.
            verbose (bool, optional): If True, prints progress information during processing.
                Defaults to True.

        Returns:
            list: List of classification result dictionaries, one per image. See classify()
                method for result structure.
        """
        metadatas = metadatas or [None] * len(images)
        results = []
        processed = 0

        if verbose:
            print(f"\nClassifying {len(images)} images ({max_workers} workers)...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.classify, img, meta): (img, meta)
                for img, meta in zip(images, metadatas)
            }

            for future in as_completed(futures):
                result = future.result()
                #print(result)
                results.append(result)

                with self.lock:
                    processed += 1
                    if verbose:
                        icon = "✓" if result['status'] == 'success' else "✗"
                        if result['status'] != 'success':
                            print(result)
                        print(f"[{processed}/{len(images)}] {icon} {result['image_name']}")

        if verbose:
            success = sum(1 for r in results if r['status'] == 'success')
            print(f"\n✓ Complete: {success}/{len(results)} successful")

        return results


def load_categories(csv_path: str) -> list:
    """Load categories from keyword CSV file.

    Reads a CSV file containing category definitions and extracts column names
    as category labels, excluding reserved columns 'Id' and 'Item'.

    Args:
        csv_path (str): Path to the categories CSV file.

    Returns:
        list: List of category names (excluding 'Id' and 'Item' columns).

    Raises:
        FileNotFoundError: If csv_path does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
    """
    df = pd.read_csv(csv_path)
    return [c for c in df.columns if c not in ['Id', 'Item']]
