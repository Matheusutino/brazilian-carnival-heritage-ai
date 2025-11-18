"""Classical image captioning script.

This script generates captions for images using classical vision-language models
such as BLIP, GIT, or ViT-GPT2. It supports optional translation to Portuguese
using machine translation models and can process images in parallel.
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.caption.caption_generator import CaptionGenerator


def parse_args():
    """Parse command-line arguments for classical image captioning.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing:
            - dataset: Path to the dataset CSV file
            - output_dir: Directory for output files
            - backend: Captioning model backend ('blip', 'git', or 'vit-gpt2')
            - device: Device to run on ('cpu' or 'cuda')
            - translate: Flag to enable Portuguese translation
            - translator_model: Translation model identifier
            - max_length: Maximum generation length for captions
            - batch_size: Number of images to process in each batch
            - use_threaded: Use threaded processing (legacy mode)
            - max_workers: Number of parallel workers (only for threaded mode)
            - limit: Maximum number of images to process
            - filter_year: Filter images by year
            - quiet: Enable quiet mode
    """
    parser = argparse.ArgumentParser(description='Generate classical captions for images')

    parser.add_argument('--dataset', default='dataset/dataset_completo.csv', help='Dataset CSV file')
    parser.add_argument('--output-dir', default='results/caption/classical', help='Output directory')
    parser.add_argument('--backend', default='blip',
                       choices=['blip', 'git', 'vit-gpt2'],
                       help='Captioning model backend')
    parser.add_argument('--device', default=None, help='Device to run on (auto-detects CUDA if available)')
    parser.add_argument('--no-translate', action='store_true',
                       help='Disable translation to Portuguese (by default, captions are translated)')
    parser.add_argument('--translator-model', default='facebook/mbart-large-50-many-to-many-mmt',
                       help='Translation model (only used when translation is enabled)')
    parser.add_argument('--max-length', type=int, default=200,
                       help='Maximum generation length')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for processing (default: 8). Larger = faster but more GPU memory')
    parser.add_argument('--use-threaded', action='store_true',
                       help='Use legacy threaded processing instead of batch processing')
    parser.add_argument('--max-workers', type=int, default=1,
                       help='Parallel workers for threaded mode (only used with --use-threaded)')
    parser.add_argument('--limit', type=int, help='Limit number of images')
    parser.add_argument('--filter-year', type=int, help='Filter by year')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode')

    return parser.parse_args()


def main():
    """Main execution function for classical image captioning.

    Orchestrates the complete caption generation pipeline:
    1. Parses command-line arguments
    2. Loads and filters the dataset
    3. Prepares image paths for processing
    4. Initializes the caption generator with specified backend
    5. Generates captions for images (optionally with translation)
    6. Saves caption results to JSON file organized by backend

    The function handles all logging and error reporting based on the quiet flag.
    Results are saved with timestamps to a backend-specific subdirectory within
    the output directory.
    """
    args = parse_args()

    # Auto-detect device if not specified
    import torch
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Translation is enabled by default (unless --no-translate is used)
    translate = not args.no_translate

    if not args.quiet:
        print("="*80)
        print("CLASSICAL CAPTION GENERATION")
        print("="*80)
        print(f"Backend: {args.backend}")
        print(f"Device: {args.device}")
        if args.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Translation: {'Enabled (EN→PT)' if translate else 'Disabled (EN only)'}")
        print(f"Mode: {'Threaded' if args.use_threaded else 'Batch'}")
        if not args.use_threaded:
            print(f"Batch size: {args.batch_size}")

    # Load dataset
    df = pd.read_csv(args.dataset)
    df = df[df['LocalPath'].notna()]

    # Filter only images (exclude videos, audio, documents, etc.)
    if 'Tipo do arquivo' in df.columns:
        original_count = len(df)
        df = df[df['Tipo do arquivo'].str.lower() == 'imagem']
        if not args.quiet and original_count != len(df):
            print(f"Filtered to images only: {len(df)} images (excluded {original_count - len(df)} non-image files)")

    # Apply filters
    if args.filter_year:
        df = df[df['Ano do desfile'] == args.filter_year]
        if not args.quiet:
            print(f"Filtered by year {args.filter_year}: {len(df)} items")

    if args.limit:
        df = df.head(args.limit)

    if not args.quiet:
        print(f"\nImages to process: {len(df)}")

    if len(df) == 0:
        print("✗ No images found")
        return

    # Prepare image paths and validate extensions
    # Accept all image extensions (case-insensitive: .jpg, .JPG, .JPEG, etc)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
    images = []
    skipped = 0

    for p in df['LocalPath']:
        path = Path(p)
        name = path.name

        # Skip hidden/macOS metadata files to stay consistent with other pipelines
        if name.startswith('._') or name.startswith('.'):
            skipped += 1
            if not args.quiet:
                print(f"⚠ Skipping hidden file: {name}")
            continue

        # Check extension case-insensitively (.jpg, .JPG, .JPEG, etc all accepted)
        if path.suffix.lower() in valid_extensions:
            images.append(path)
        else:
            skipped += 1
            if not args.quiet:
                print(f"⚠ Skipping non-image file: {path.name} (extension: {path.suffix})")

    if not args.quiet and skipped > 0:
        print(f"\nSkipped {skipped} files (hidden or unsupported)")

    # Generate captions
    generator = CaptionGenerator(
        backend=args.backend,
        device=args.device,
        translate=translate,
        translator_model=args.translator_model if translate else None,
        max_length=args.max_length
    )

    # Use batch or threaded processing
    if args.use_threaded:
        results = generator.generate_batch_threaded(images, args.max_workers, not args.quiet)
    else:
        results = generator.generate_batch(images, args.batch_size, not args.quiet)

    # Save results
    backend_dir = Path(args.output_dir) / args.backend
    backend_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = backend_dir / f"captions_{timestamp}.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')

    if not args.quiet:
        success = sum(1 for r in results if r['status'] == 'success')
        print(f"\n✓ Saved: {json_path}")
        print(f"✓ Success: {success}/{len(results)}")


if __name__ == "__main__":
    main()
