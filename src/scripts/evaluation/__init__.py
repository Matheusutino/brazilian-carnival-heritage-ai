"""Evaluation scripts for assessing processing results.

This package provides CLI scripts to evaluate and measure the quality of
pipeline outputs, including:
- Caption similarity evaluation
- LLM-based classification evaluation
- Supervised classification evaluation
- Image upscaling quality assessment

Each script exposes a standalone CLI and reads inputs from the project
conventions (e.g., CSVs under `dataset/` and results under `outputs/`).
"""

__all__ = [
    "evaluate_caption_similarity",
    "evaluate_caption_outputs",
    "evaluate_llm_classifications",
    "evaluate_classifications",
    "evaluate_upscale_quality",
]

__version__ = "1.0.0"
