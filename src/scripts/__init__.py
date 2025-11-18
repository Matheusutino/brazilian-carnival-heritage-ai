"""Image Classification and Captioning Scripts.

This module contains command-line scripts for performing various image analysis tasks
including classification and caption generation using different approaches:

Scripts:
    - llm_classify_images.py: Classify images using multimodal Large Language Models
    - zero_shot_classify_images.py: Zero-shot classification using CLIP/SigLIP models
    - caption_images_classical.py: Generate captions using classical vision models
    - prepare_dataset.py: Orchestrate the complete dataset preparation pipeline

Each script provides a standalone CLI interface with configurable parameters for
model selection, batch processing, filtering, and output management.

Example:
    Run LLM-based classification:
        $ python -m src.scripts.llm_classify_images --model meta/llama-4 --limit 100

    Run zero-shot classification:
        $ python -m src.scripts.zero_shot_classify_images --model google/siglip

    Generate classical captions:
        $ python -m src.scripts.caption_images_classical --backend blip --translate
"""

__version__ = "1.0.0"
