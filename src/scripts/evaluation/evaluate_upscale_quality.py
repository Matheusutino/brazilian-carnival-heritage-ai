"""Evaluate upscaled image quality using comprehensive metrics.

This script evaluates upscaled images using various quality metrics including:
- Sharpness (Variance of Laplacian)
- Edge Strength (Sobel operators)
- Contrast (standard deviation)
- Artifacts (over-sharpening, ringing)
- Self-consistency (PSNR with downsampled version)
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.evaluation.upscale_metrics import ImageQualityEvaluator


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate upscaled image quality'
    )

    parser.add_argument('--upscaled-dir', required=True,
                       help='Directory containing upscaled images')
    parser.add_argument('--original-dir', default='dataset/downloaded',
                       help='Directory containing original images (for self-consistency)')
    parser.add_argument('--output-dir', default='results/upscale',
                       help='Output directory for metrics (default: results/upscale_evaluation)')
    parser.add_argument('--n-workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')

    return parser.parse_args()


METRIC_LABELS = {
    'sharpness': 'Sharpness',
    'edge_strength': 'Edge Strength',
    'contrast': 'Contrast',
    'oversharp': 'Over-sharpening',
    'ringing': 'Ringing',
    'self_consistency': 'Self-Consistency (PSNR)'
}


def compute_aggregate_metrics(results):
    """Compute aggregate statistics for successful evaluations."""
    successful = [r for r in results if r['status'] == 'success']

    summary = {
        'total_images': len(results),
        'successful_images': len(successful),
        'metrics': {}
    }

    if not successful:
        return summary

    for key in METRIC_LABELS:
        values = [
            r[key] for r in successful
            if key in r and r[key] is not None
        ]
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        summary['metrics'][key] = {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr))
        }

    return summary


def print_metrics_summary(results):
    """Print formatted metrics summary.

    Args:
        results (list): List of evaluation result dictionaries.
    """
    summary = compute_aggregate_metrics(results)

    if summary['successful_images'] == 0:
        print("\n✗ No successful evaluations")
        return

    print("\n" + "="*80)
    print("UPSCALE QUALITY METRICS SUMMARY")
    print("="*80)
    print(f"\nImages evaluated: {summary['successful_images']}/{summary['total_images']}")

    print(f"\n{'Metric':<25} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-"*80)

    if not summary['metrics']:
        print("No metric values available.")
        return

    for key, stats in summary['metrics'].items():
        label = METRIC_LABELS.get(key, key)
        print(f"{label:<25} "
              f"{stats['mean']:<12.2f} "
              f"{stats['std']:<12.2f} "
              f"{stats['min']:<12.2f} "
              f"{stats['max']:<12.2f}")


def main():
    """Main execution function."""
    args = parse_args()

    upscaled_dir = Path(args.upscaled_dir)
    original_dir = Path(args.original_dir)

    if not args.quiet:
        print("="*80)
        print("UPSCALE QUALITY EVALUATION")
        print("="*80)
        print(f"Upscaled images: {upscaled_dir}")
        print(f"Original images: {original_dir}")

    if not upscaled_dir.exists():
        print(f"\n✗ Error: Upscaled directory not found: {upscaled_dir}")
        return

    # Get all images in upscaled directory
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(upscaled_dir.glob(f"*{ext}"))
        image_paths.extend(upscaled_dir.glob(f"*{ext.upper()}"))

    # Filter out hidden files
    image_paths = [p for p in image_paths if not p.name.startswith('.')]
    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"\n✗ Error: No images found in {upscaled_dir}")
        return

    if not args.quiet:
        print(f"\nImages to evaluate: {len(image_paths)}")

    # Initialize evaluator
    evaluator = ImageQualityEvaluator(original_images_dir=original_dir)

    # Evaluate images
    results = evaluator.evaluate_batch(
        image_paths=image_paths,
        verbose=not args.quiet,
        n_workers=args.n_workers
    )

    summary = compute_aggregate_metrics(results)

    # Print summary
    if not args.quiet:
        print_metrics_summary(results)

    # Save to file
    # Extract method name from upscaled directory (e.g., "bicubic_2x" from path)
    method_name = upscaled_dir.name

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{method_name}_metrics.json"

    output_data = {
        'timestamp': datetime.now().isoformat(),
        'method': method_name,
        'upscaled_dir': str(upscaled_dir),
        'original_dir': str(original_dir),
        'n_images': summary['total_images'],
        'n_successful': summary['successful_images'],
        'aggregated_metrics': summary['metrics'],
        'results': results
    }

    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding='utf-8')

    if not args.quiet:
        print(f"\n✓ Metrics saved: {output_path}")

    if not args.quiet:
        print("\n" + "="*80)
        print("✓ Evaluation completed!")
        print("="*80)


if __name__ == "__main__":
    main()
