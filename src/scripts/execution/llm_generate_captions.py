"""Image caption generation script using multimodal LLM.

This script generates rich, structured captions for images using large language models
with vision capabilities. It produces detailed descriptions including visual elements,
cultural analysis, temporal context, and technical quality assessment.
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.caption.llm_captioner import LLMCaptioner


def parse_args():
    """Parse command-line arguments for LLM-based caption generation.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing:
            - dataset: Path to the dataset CSV file
            - config: Path to the prompts configuration YAML file
            - output_dir: Directory for output files
            - provider: LLM provider (e.g., 'nvidia', 'openai')
            - model: Model name/identifier
            - prompt_key: Key in config YAML for prompts
            - temperature: Initial temperature for generation
            - max_tokens: Maximum number of output tokens
            - max_workers: Number of parallel workers
            - max_retries: Maximum retry attempts for failed generations
            - retry_temp_min: Minimum temperature for retry attempts
            - retry_temp_max: Maximum temperature for retry attempts
            - limit: Maximum number of images to process
            - filter_year: Filter images by year
            - quiet: Enable quiet mode
    """
    parser = argparse.ArgumentParser(description='Generate image captions with LLM')

    # Data
    parser.add_argument('--dataset', default='dataset/dataset_completo.csv', help='Dataset CSV file')
    parser.add_argument('--config', default='config/prompts.yaml', help='Configuration YAML file')
    parser.add_argument('--output-dir', default='results/captions/llm', help='Output directory')

    # Model
    parser.add_argument('--provider', default='nvidia', help='LLM provider')
    parser.add_argument('--model', default='mistralai/mistral-small-3.1-24b-instruct-2503', help='Model name')
    parser.add_argument('--prompt-key', default='caption', help='Prompt key in config YAML')
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
    parser.add_argument('--limit', type=int, help='Limit number of images')
    parser.add_argument('--filter-year', type=int, help='Filter by year')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode')

    return parser.parse_args()


def main():
    """Main execution function for LLM-based image caption generation.

    Orchestrates the complete caption generation pipeline:
    1. Parses command-line arguments
    2. Loads dataset
    3. Applies filters (year, limit) to the dataset
    4. Prepares image paths and metadata
    5. Initializes the LLM captioner with retry configuration
    6. Processes images in parallel using multiple workers
    7. Saves caption results to JSON file organized by model

    The function handles all logging and error reporting based on the quiet flag.
    Results are saved with timestamps to a model-specific subdirectory within
    the output directory.
    """
    args = parse_args()

    if not args.quiet:
        print("="*80)
        print("IMAGE CAPTION GENERATION")
        print("="*80)
        print(f"Config: {args.config}")
        print(f"Dataset: {args.dataset}")
        print(f"Workers: {args.max_workers}")

    # Load dataset
    df = pd.read_csv(args.dataset)
    df = df[df['LocalPath'].notna()]

    # Filter only images by file extension (exclude videos, audio, documents, etc.)
    # Accept all image extensions case-insensitively (.jpg, .JPG, .JPEG, etc)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
    df = df[df['LocalPath'].str.lower().str.endswith(tuple(valid_extensions))]

    # Apply filters
    if args.filter_year:
        df = df[df['Ano do desfile'] == args.filter_year]
        if not args.quiet:
            print(f"Filtered by year {args.filter_year}: {len(df)} items")

    if args.limit:
        df = df.head(args.limit)

    # Prepare data - filter out macOS metadata files (._*) and hidden files
    images = [Path(p) for p in df['LocalPath'] if not Path(p).name.startswith('._') and not Path(p).name.startswith('.')]
    metadatas = [
        {
            'id': row.get('Id'),
            'item': row.get('Item'),
            'ano': row.get('Ano do desfile')
        }
        for _, row in df.iterrows()
        if not Path(row['LocalPath']).name.startswith('._') and not Path(row['LocalPath']).name.startswith('.')
    ]

    if not args.quiet:
        print(f"\nImages to process: {len(images)}")

    if len(images) == 0:
        print("✗ No images found")
        return

    # Generate captions
    captioner = LLMCaptioner(
        args.config,
        args.provider, args.model,
        args.temperature, args.max_tokens,
        args.max_retries,
        args.retry_temp_min, args.retry_temp_max,
        args.prompt_key,
        max_image_dim=args.max_image_dim
    )
    results = captioner.generate_batch(images, metadatas, args.max_workers, not args.quiet)

    # Save results to output-dir/{model}/{prompt_name}/
    prompt_name = args.prompt_key.split('/')[-1] if '/' in args.prompt_key else args.prompt_key
    model_dir = Path(args.output_dir) / args.model.replace('/', '_') / prompt_name
    model_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = model_dir / f"captions_{timestamp}.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')

    if not args.quiet:
        success = sum(1 for r in results if r['status'] == 'success')
        total_attempts = sum(len(r.get('attempts', [])) for r in results)
        print(f"\n✓ Saved: {json_path}")
        print(f"✓ Success: {success}/{len(results)}")
        print(f"✓ Total attempts: {total_attempts}")


if __name__ == "__main__":
    main()
