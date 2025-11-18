"""Image caption generation module using classical vision models."""
from pathlib import Path
from typing import List, Dict
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import torch
from tqdm import tqdm


class CaptionGenerator:
    """Image caption generator using classical vision models."""

    def __init__(self, backend: str = 'blip-pt', device: str = 'cuda', translate: bool = True,
                 translator_model: str = 'facebook/mbart-large-50-many-to-many-mmt',
                 max_length: int = 200):
        """Initialize caption generator.

        Args:
            backend: Captioning model backend ('blip', 'git', or 'vit-gpt2').
            device: Device to run on ('cpu' or 'cuda').
            translate: If True, translates captions to Portuguese.
            translator_model: Translation model name.
            max_length: Maximum generation length.
        """
        self.backend = backend
        self.device = device
        self.translate = translate
        self.translator_model_name = translator_model
        self.max_length = max_length
        self.model = None
        self.processor = None
        self.translator_model = None
        self.translator_tokenizer = None
        self.lock = Lock()
        self._load_model()
        if self.translate:
            self._load_translator()

    def _load_model(self):
        """Load the vision model based on backend configuration.

        This method initializes the appropriate vision-language model based on the
        specified backend. Supported backends include BLIP, GIT, and ViT-GPT2,
        each with different architectures and capabilities.
        """
        print(f"Loading model: {self.backend}...")

        # BLIP (Bootstrapping Language-Image Pre-training)
        if self.backend == 'blip':
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model.to(self.device)

        # GIT (Generative Image-to-text Transformer)
        elif self.backend == 'git':
            from transformers import AutoProcessor, AutoModelForCausalLM
            self.processor = AutoProcessor.from_pretrained("microsoft/git-base")
            self.model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
            self.model.to(self.device)

        # ViT-GPT2 (Vision Transformer encoder + GPT-2 decoder)
        elif self.backend == 'vit-gpt2':
            from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
            self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.model.to(self.device)

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        print("✓ Model loaded")

    def _load_translator(self):
        """Load mBART translation model for English-to-Portuguese translation.

        mBART (Multilingual BART) is a sequence-to-sequence model trained on multiple
        languages, capable of high-quality machine translation.
        """
        print(f"Loading translator: {self.translator_model_name}...")
        from transformers import MBartForConditionalGeneration, MBart50Tokenizer

        self.translator_model = MBartForConditionalGeneration.from_pretrained(
            self.translator_model_name
        )
        self.translator_tokenizer = MBart50Tokenizer.from_pretrained(
            self.translator_model_name
        )
        self.translator_model.to(self.device)
        print("✓ Translator loaded")

    def _translate_to_pt(self, text_en: str) -> str:
        """Translate English text to Portuguese.

        Args:
            text_en: English text to translate.

        Returns:
            Translated Portuguese text.
        """
        if not self.translator_model:
            return text_en

        # Set English as source language
        self.translator_tokenizer.src_lang = "en_XX"

        # Tokenize the input English text
        encoded = self.translator_tokenizer(text_en, return_tensors="pt")
        # Move tensors to the appropriate device (CPU or GPU)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Generate translation forcing Portuguese as target language
        # forced_bos_token_id ensures the model generates Portuguese output
        generated_tokens = self.translator_model.generate(
            **encoded,
            forced_bos_token_id=self.translator_tokenizer.lang_code_to_id["pt_XX"],
            max_length=self.max_length
        )

        # Decode tokens back to text
        text_pt = self.translator_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return text_pt

    def _translate_batch_to_pt(self, texts_en: List[str]) -> List[str]:
        """Translate batch of English texts to Portuguese.

        Args:
            texts_en: List of English texts to translate.

        Returns:
            List of translated Portuguese texts.
        """
        if not self.translator_model or not texts_en:
            return texts_en

        # Set English as source language
        self.translator_tokenizer.src_lang = "en_XX"

        # Tokenize all texts
        encoded = self.translator_tokenizer(texts_en, return_tensors="pt", padding=True, truncation=True)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Generate translations in batch
        generated_tokens = self.translator_model.generate(
            **encoded,
            forced_bos_token_id=self.translator_tokenizer.lang_code_to_id["pt_XX"],
            max_length=self.max_length
        )

        # Decode all tokens
        texts_pt = self.translator_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return texts_pt

    def _generate_captions_batch_internal(self, images: List[Image.Image]) -> List[str]:
        """Generate captions for a batch of images.

        Args:
            images: List of PIL Image objects.

        Returns:
            List of generated captions (in English).
        """
        if not images:
            return []

        with torch.no_grad():
            if self.backend == 'vit-gpt2':
                # Process batch of images
                pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)

                # Generate captions for batch
                outputs = self.model.generate(pixel_values, max_length=self.max_length)

                # Decode all captions
                captions = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            else:
                # BLIP/GIT batch processing
                inputs = self.processor(images=images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate captions for batch
                outputs = self.model.generate(**inputs, max_length=self.max_length)

                # Decode all captions
                captions = [self.processor.decode(output, skip_special_tokens=True) for output in outputs]

        return captions

    def generate_caption(self, image_path: Path) -> Dict:
        """Generate caption for a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            Dictionary containing caption_en, caption_pt (if translate=True),
            status, and other metadata.
        """
        result = {
            'image_path': str(image_path),
            'image_name': image_path.name,
            'backend': self.backend,
            'status': 'pending'
        }

        try:
            # Open and convert image to RGB format
            image = Image.open(image_path).convert('RGB')

            # Generate caption based on backend
            if self.backend == 'vit-gpt2':
                # ViT-GPT2 has a different processing pipeline
                pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)
                # Generate caption tokens
                outputs = self.model.generate(pixel_values, max_length=self.max_length)
                # Decode tokens to text
                caption_en = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                # BLIP/GIT use similar pipelines
                inputs = self.processor(images=image, return_tensors="pt")
                # Move all input tensors to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                # Generate caption tokens
                outputs = self.model.generate(**inputs, max_length=self.max_length)
                # Decode tokens to text
                caption_en = self.processor.decode(outputs[0], skip_special_tokens=True)

            result['caption_en'] = caption_en

            # Translate to Portuguese if requested
            if self.translate:
                caption_pt = self._translate_to_pt(caption_en)
                result['caption_pt'] = caption_pt
                result['caption'] = caption_pt  # Main caption in Portuguese
            else:
                result['caption'] = caption_en  # Main caption in English

            result['status'] = 'success'

        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    def generate_batch(self, image_paths: List[Path], batch_size: int = 8, verbose: bool = True) -> List[Dict]:
        """Generate captions for multiple images using efficient batch processing.

        This method processes images in batches using native PyTorch batch processing,
        which is much more efficient than processing images one at a time, especially
        on CUDA devices.

        Args:
            image_paths: List of paths to image files.
            batch_size: Number of images to process in each batch (default: 8).
                       Larger batches are faster but use more GPU memory.
            verbose: If True, prints progress information.

        Returns:
            List of dictionaries containing caption results, in the same order as input.
        """
        results = []

        if verbose:
            print(f"\nGenerating captions for {len(image_paths)} images...")
            print(f"Batch size: {batch_size}, Device: {self.device}")

        # Process images in batches
        num_batches = (len(image_paths) + batch_size - 1) // batch_size

        iterator = range(0, len(image_paths), batch_size)
        if verbose:
            iterator = tqdm(iterator, total=num_batches, desc="Processing batches")

        for i in iterator:
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            batch_valid_indices = []

            # Load images for this batch
            for idx, img_path in enumerate(batch_paths):
                try:
                    image = Image.open(img_path).convert('RGB')
                    batch_images.append(image)
                    batch_valid_indices.append(idx)
                except Exception as e:
                    # Record error for this image
                    results.append({
                        'image_path': str(img_path),
                        'image_name': img_path.name,
                        'backend': self.backend,
                        'status': 'error',
                        'error': f"Failed to load image: {str(e)}"
                    })

            # Generate captions for valid images in batch
            if batch_images:
                try:
                    captions_en = self._generate_captions_batch_internal(batch_images)

                    # Translate batch if needed
                    if self.translate:
                        captions_pt = self._translate_batch_to_pt(captions_en)
                    else:
                        captions_pt = None

                    # Build results for this batch
                    for idx, (img_path, caption_en) in enumerate(zip(
                        [batch_paths[i] for i in batch_valid_indices], captions_en
                    )):
                        result = {
                            'image_path': str(img_path),
                            'image_name': img_path.name,
                            'backend': self.backend,
                            'caption_en': caption_en,
                            'status': 'success'
                        }

                        if self.translate and captions_pt:
                            result['caption_pt'] = captions_pt[idx]
                            result['caption'] = captions_pt[idx]
                        else:
                            result['caption'] = caption_en

                        results.append(result)

                except Exception as e:
                    # If batch processing fails, record error for all images in batch
                    for img_path in [batch_paths[i] for i in batch_valid_indices]:
                        results.append({
                            'image_path': str(img_path),
                            'image_name': img_path.name,
                            'backend': self.backend,
                            'status': 'error',
                            'error': f"Batch processing failed: {str(e)}"
                        })

        if verbose:
            success = sum(1 for r in results if r['status'] == 'success')
            print(f"\n✓ Complete: {success}/{len(results)} successful")

        return results

    def generate_batch_threaded(self, image_paths: List[Path], max_workers: int = 4, verbose: bool = True) -> List[Dict]:
        """Generate captions for multiple images using thread-based parallelism (legacy).

        This method is kept for compatibility but is less efficient than the native
        batch processing in generate_batch(). Use generate_batch() instead for better
        performance, especially on CUDA.

        Args:
            image_paths: List of paths to image files.
            max_workers: Number of parallel workers.
            verbose: If True, prints progress information.

        Returns:
            List of dictionaries containing caption results.
        """
        results = []
        processed = 0

        if verbose:
            print(f"\nGenerating captions for {len(image_paths)} images (threaded mode)...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.generate_caption, img): img for img in image_paths}

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                with self.lock:
                    processed += 1
                    if verbose:
                        icon = "✓" if result['status'] == 'success' else "✗"
                        print(f"[{processed}/{len(image_paths)}] {icon} {result['image_name']}")

        if verbose:
            success = sum(1 for r in results if r['status'] == 'success')
            print(f"\n✓ Complete: {success}/{len(results)} successful")

        return results
