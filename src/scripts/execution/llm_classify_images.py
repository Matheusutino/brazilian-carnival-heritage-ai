"""Image classification script using multimodal LLM.

This script performs image classification using large language models with vision
capabilities. It uses configurable prompts to guide the LLM in classifying images
into predefined categories, with support for retry mechanisms and parallel processing.
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.classification.llm_classifier import ImageClassifier, load_categories
from src.core.data.dataset_loader import MultilabelDataset


def parse_args():
    """Parse command-line arguments for LLM-based classification.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing:
            - split: Path to train/test split JSON file
            - config: Path to the prompts configuration YAML file
            - categories: Path to the categories CSV file
            - output_dir: Directory for output files
            - provider: LLM provider (e.g., 'nvidia', 'openai')
            - model: Model name/identifier
            - prompt_key: Key path in config YAML for prompts
            - temperature: Initial temperature for generation
            - max_tokens: Maximum number of output tokens
            - max_workers: Number of parallel workers
            - max_retries: Maximum retry attempts for failed classifications
            - retry_temp_min: Minimum temperature for retry attempts
            - retry_temp_max: Maximum temperature for retry attempts
            - quiet: Enable quiet mode
    """
    parser = argparse.ArgumentParser(description='Classify images with LLM')

    # Data
    parser.add_argument('--split', default='dataset/splits/split_latest.json', help='Train/test split JSON file')
    parser.add_argument('--config', default='config/prompts.yaml', help='Configuration YAML file')
    parser.add_argument('--categories', default='dataset/palavras_chave_processed.csv', help='Categories CSV file')
    parser.add_argument('--output-dir', default='results/classification/llm', help='Output directory')

    # Model
    parser.add_argument('--provider', default='nvidia', help='LLM provider')
    parser.add_argument('--model', default='mistralai/mistral-small-3.1-24b-instruct-2503', help='Model name')
    parser.add_argument('--prompt-key', default='classification/zero_shot_basic', help='Prompt key in config YAML')
    parser.add_argument('--temperature', type=float, default=0.0, help='Initial temperature')
    parser.add_argument('--max-tokens', type=int, default=8000, help='Maximum output tokens')

    # Processing
    parser.add_argument('--max-workers', type=int, default=4, help='Parallel workers')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum retry attempts')
    parser.add_argument('--retry-temp-min', type=float, default=0.0,
                       help='Minimum temperature for retry attempts')
    parser.add_argument('--retry-temp-max', type=float, default=1.0,
                       help='Maximum temperature for retry attempts')
    parser.add_argument('--max-image-dim', type=int,
                       help='Resize images to this maximum dimension before sending to the LLM (optional)')
    parser.add_argument('--limit', type=int,
                       help='Limit number of test images to process (takes first N images)')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode')

    return parser.parse_args()


def main():
    """Main execution function for LLM-based image classification.

    Orchestrates the complete LLM classification pipeline:
    1. Parses command-line arguments
    2. Loads categories and test set from split
    3. Prepares image paths and metadata
    4. Initializes the LLM classifier with retry configuration
    5. Processes images in parallel using multiple workers
    6. Saves classification results to JSON file organized by model

    The function handles all logging and error reporting based on the quiet flag.
    Results are saved with timestamps to a model-specific subdirectory within
    the output directory.
    """
    args = parse_args()

    if not args.quiet:
        print("="*80)
        print("IMAGE CLASSIFICATION - TEST SET ONLY")
        print("="*80)
        print(f"Config: {args.config}")
        print(f"Split: {args.split}")
        print(f"Workers: {args.max_workers}")

    # Load categories
    categories = load_categories(args.categories)
    if not args.quiet:
        print(f"\nCategories: {len(categories)}")

    # Load test set from split
    dataset = MultilabelDataset(args.split, labels_source=None)
    test_paths, _ = dataset.get_test_data()

    # Apply limit if specified
    if args.limit is not None and args.limit > 0:
        test_paths = test_paths[:args.limit]

    if not args.quiet:
        print(f"\nTest images to process: {len(test_paths)}")

    if len(test_paths) == 0:
        print("✗ No test images found")
        return

    # Prepare data
    images = test_paths
    metadatas = [{'path': str(p)} for p in test_paths]

    # Classify
    classifier = ImageClassifier(
        args.config, categories,
        args.provider, args.model,
        args.temperature, args.max_tokens,
        args.max_retries,
        args.retry_temp_min, args.retry_temp_max,
        args.prompt_key,
        max_image_dim=args.max_image_dim
    )
    results = classifier.classify_batch(images, metadatas, args.max_workers, not args.quiet)

    # Save results to output-dir/{model}/{prompt_name}/
    prompt_name = args.prompt_key.split('/')[-1]  # Extract last part of prompt key (e.g., 'zero_shot_basic')
    model_dir = Path(args.output_dir) / args.model.replace('/', '_') / prompt_name
    model_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = model_dir / f"classifications_{timestamp}.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')

    if not args.quiet:
        success = sum(1 for r in results if r['status'] == 'success')
        total_attempts = sum(len(r.get('attempts', [])) for r in results)
        print(f"\n✓ Saved: {json_path}")
        print(f"✓ Success: {success}/{len(results)}")
        print(f"✓ Total attempts: {total_attempts}")


if __name__ == "__main__":
    main()
