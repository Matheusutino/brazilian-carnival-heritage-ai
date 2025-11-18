"""Image upscaling script using Real-ESRGAN, ESRGAN, and EDSR.

This script upscales images using state-of-the-art super-resolution models.
It supports Real-ESRGAN, ESRGAN, and EDSR with 2x and 4x upscaling factors.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.upscaler.image_upscaler import ImageUpscaler


def parse_args():
    """Parse command-line arguments for image upscaling.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing:
            - dataset: Path to the dataset CSV file (optional)
            - input_dir: Directory containing input images
            - output_dir: Directory for upscaled images
            - model: Upscaling model ('realesrgan', 'esrgan', or 'edsr')
            - scale: Upscaling factor (2 or 4)
            - device: Device to run on ('cpu' or 'cuda')
            - limit: Maximum number of images to process
            - filter_year: Filter images by year (requires dataset)
            - quiet: Enable quiet mode
    """
    parser = argparse.ArgumentParser(
        description='Upscale images using deep learning models or classical interpolation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deep learning models (Real-ESRGAN, ESRGAN, EDSR)
  python src/scripts/upscale_images.py --input-dir dataset/downloaded --model realesrgan --scale 4
  python src/scripts/upscale_images.py --dataset dataset/dataset_completo.csv --model edsr --scale 2
  python src/scripts/upscale_images.py --input-dir dataset/downloaded --model esrgan --scale 4 --device cpu

  # Classical interpolation methods (fast, no GPU required)
  python src/scripts/upscale_images.py --input-dir dataset/downloaded --model bicubic --scale 2
  python src/scripts/upscale_images.py --input-dir dataset/downloaded --model lanczos --scale 4
  python src/scripts/upscale_images.py --input-dir dataset/downloaded --model bilinear --scale 3

  # Limit number of images for testing
  python src/scripts/upscale_images.py --input-dir dataset/downloaded --model nearest --scale 2 --limit 10
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--dataset', help='Dataset CSV file (uses LocalPath column)')
    input_group.add_argument('--input-dir', help='Directory containing input images')

    # Output
    parser.add_argument('--output-dir', default='dataset/upscaled', help='Output directory for upscaled images')

    # Model configuration
    parser.add_argument('--model', default='bicubic',
                       choices=['realesrgan', 'esrgan', 'edsr', 'nearest', 'bilinear', 'bicubic', 'lanczos'],
                       help='Upscaling method (default: bicubic). '
                            'Deep learning: realesrgan, esrgan, edsr (scales: 2, 4). '
                            'Classical: nearest, bilinear, bicubic, lanczos (any scale)')
    parser.add_argument('--scale', type=int, default=2,
                       help='Upscaling factor (default: 2). '
                            'Deep learning models support 2 or 4. Classical methods support any integer.')
    parser.add_argument('--tile-size', type=int,
                        help='Tile size for tiled inference (default: auto). '
                             'Use 0 to disable tiling entirely.')
    parser.add_argument('--tile-pad', type=int, default=10,
                        help='Tile padding (in pixels) used between tiles when tiling is enabled.')

    # Device
    parser.add_argument('--device', default=None,
                       help='Device to run on (auto-detects CUDA if available)')

    # Filtering
    parser.add_argument('--limit', type=int,
                       help='Limit number of images to process')
    parser.add_argument('--filter-year', type=int,
                       help='Filter by year (only when using --dataset)')

    # Output control
    parser.add_argument('--quiet', action='store_true',
                       help='Quiet mode (minimal output)')

    args = parser.parse_args()

    # Validate filter-year usage
    if args.filter_year and not args.dataset:
        parser.error("--filter-year requires --dataset")

    return args


def get_image_paths_from_dataset(dataset_path: str, filter_year: int = None, limit: int = None) -> list[Path]:
    """Load image paths from dataset CSV.

    Args:
        dataset_path: Path to the dataset CSV file.
        filter_year: Optional year filter.
        limit: Optional limit on number of images.

    Returns:
        List of image paths.
    """
    df = pd.read_csv(dataset_path)
    df = df[df['LocalPath'].notna()]

    # Filter only images by file extension (exclude videos, audio, documents, etc.)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
    df = df[df['LocalPath'].str.lower().str.endswith(tuple(valid_extensions))]

    # Apply year filter
    if filter_year and 'Ano do desfile' in df.columns:
        df = df[df['Ano do desfile'] == filter_year]

    # Apply limit
    if limit:
        df = df.head(limit)

    # Extract paths and filter out macOS metadata files (._*) and hidden files
    image_paths = [Path(p) for p in df['LocalPath'] if not Path(p).name.startswith('._') and not Path(p).name.startswith('.')]

    return image_paths


def get_image_paths_from_directory(input_dir: str, limit: int = None) -> list[Path]:
    """Load image paths from a directory.

    Args:
        input_dir: Directory containing images.
        limit: Optional limit on number of images.

    Returns:
        List of image paths.
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    # Valid image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}

    # Collect all image files
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(input_path.glob(f"*{ext}"))
        image_paths.extend(input_path.glob(f"*{ext.upper()}"))

    # Filter out macOS metadata files (._*) and hidden files
    image_paths = [p for p in image_paths if not p.name.startswith('._') and not p.name.startswith('.')]

    # Sort for consistent ordering
    image_paths = sorted(image_paths)

    # Apply limit
    if limit:
        image_paths = image_paths[:limit]

    return image_paths


def main():
    """Main execution function for image upscaling.

    Orchestrates the complete upscaling pipeline:
    1. Parses command-line arguments
    2. Loads image paths from dataset or directory
    3. Initializes the upscaler with specified model and scale
    4. Upscales all images
    5. Saves results to JSON file with metadata
    """
    args = parse_args()

    # Validate scale for deep learning models
    deep_learning_models = ['realesrgan', 'esrgan', 'edsr']
    classical_methods = ['nearest', 'bilinear', 'bicubic', 'lanczos']

    if args.model in deep_learning_models and args.scale not in [2, 4]:
        print(f"✗ Error: {args.model} only supports scale 2 or 4, got {args.scale}")
        return

    # Auto-detect device if not specified (only for deep learning models)
    import torch
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not args.quiet:
        print("=" * 80)
        print("IMAGE UPSCALING")
        print("=" * 80)
        if args.model in classical_methods:
            print(f"Method: {args.model} (classical interpolation)")
        else:
            print(f"Model: {args.model}")
        print(f"Scale: {args.scale}x")

        # Only show device info for deep learning models
        if args.model in deep_learning_models:
            print(f"Device: {args.device}")
            if args.device == 'cuda':
                print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load image paths
    if args.dataset:
        if not args.quiet:
            print(f"\nLoading images from dataset: {args.dataset}")
        image_paths = get_image_paths_from_dataset(args.dataset, args.filter_year, args.limit)
        if args.filter_year and not args.quiet:
            print(f"Filtered by year: {args.filter_year}")
    else:
        if not args.quiet:
            print(f"\nLoading images from directory: {args.input_dir}")
        image_paths = get_image_paths_from_directory(args.input_dir, args.limit)

    if not args.quiet:
        print(f"Images to process: {len(image_paths)}")

    if len(image_paths) == 0:
        print("✗ No images found")
        return

    # Create output directory with model and scale info
    output_dir = Path(args.output_dir) / f"{args.model}_{args.scale}x"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"Output directory: {output_dir}")

    # Initialize upscaler
    upscaler = ImageUpscaler(
        model=args.model,
        scale=args.scale,
        device=args.device,
        tile_size=args.tile_size,
        tile_pad=args.tile_pad
    )

    # Upscale images
    results = upscaler.upscale_batch(
        image_paths=image_paths,
        output_dir=output_dir,
        verbose=not args.quiet
    )

    # Print summary
    if not args.quiet:
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'error')
        print(f"\n✓ Upscaled images saved to: {output_dir}")
        print(f"✓ Success: {successful}/{len(results)}")
        if failed > 0:
            print(f"✗ Failed: {failed}/{len(results)}")


if __name__ == "__main__":
    main()
