"""Evaluate classification predictions using comprehensive metrics.

This script loads prediction files (npz format) and computes detailed metrics
using the classification_metrics module. It supports both supervised classifiers
and LLM-based predictions.
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.evaluation.classification_metrics import (
    calculate_multilabel_metrics,
    calculate_per_image_metrics
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate classification predictions'
    )

    parser.add_argument('--predictions', required=True,
                       help='Path to predictions NPZ file')
    parser.add_argument('--output', default=None,
                       help='Output JSON file for metrics (optional)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')

    return parser.parse_args()


def load_predictions(predictions_path):
    """Load predictions from NPZ file.

    Args:
        predictions_path (str): Path to NPZ file containing y_true, y_pred, categories.

    Returns:
        tuple: (y_true, y_pred, categories, test_paths)
    """
    data = np.load(predictions_path, allow_pickle=True)

    y_true = data['y_true']
    y_pred = data['y_pred']
    categories = data['categories'].tolist() if 'categories' in data else None
    test_paths = data['test_paths'].tolist() if 'test_paths' in data else None

    return y_true, y_pred, categories, test_paths


def convert_to_dict_format(y_array, categories):
    """Convert array format (n_samples, n_labels) to dict format.

    Args:
        y_array (np.ndarray): Array of shape (n_samples, n_labels).
        categories (list): List of category names.

    Returns:
        dict: Dictionary mapping category names to lists of labels.
    """
    return {cat: y_array[:, i].tolist() for i, cat in enumerate(categories)}


def print_metrics(metrics, per_image_metrics):
    """Print formatted metrics summary."""
    print("\n" + "="*80)
    print("EVALUATION METRICS")
    print("="*80)

    # Per-image metrics
    print("\nPer-Image Metrics:")
    print(f"  Exact Match Ratio:  {per_image_metrics['exact_match_ratio']:.4f}")
    print(f"  Hamming Loss:       {per_image_metrics['hamming_loss']:.4f}")
    print(f"  Jaccard Score:      {per_image_metrics['jaccard_score']:.4f}")

    # Aggregate metrics
    print("\nMicro-averaged (overall performance):")
    print(f"  Precision:          {metrics['micro_avg']['precision']:.4f}")
    print(f"  Recall:             {metrics['micro_avg']['recall']:.4f}")
    print(f"  F1 Score:           {metrics['micro_avg']['f1']:.4f}")
    print(f"  Accuracy:           {metrics['micro_avg']['accuracy']:.4f}")

    print("\nMacro-averaged (per-class average):")
    print(f"  Precision:          {metrics['macro_avg']['precision']:.4f}")
    print(f"  Recall:             {metrics['macro_avg']['recall']:.4f}")
    print(f"  F1 Score:           {metrics['macro_avg']['f1']:.4f}")
    print(f"  Accuracy:           {metrics['macro_avg']['accuracy']:.4f}")

    print("\nWeighted-averaged (weighted by support):")
    print(f"  Precision:          {metrics['weighted_avg']['precision']:.4f}")
    print(f"  Recall:             {metrics['weighted_avg']['recall']:.4f}")
    print(f"  F1 Score:           {metrics['weighted_avg']['f1']:.4f}")
    print(f"  Accuracy:           {metrics['weighted_avg']['accuracy']:.4f}")

    # Per-category metrics (top 5 by F1)
    categories = [k for k in metrics.keys() if k not in ['micro_avg', 'macro_avg', 'weighted_avg']]
    sorted_cats = sorted(categories, key=lambda c: metrics[c]['f1'], reverse=True)

    print("\nTop 5 categories by F1:")
    for cat in sorted_cats[:5]:
        m = metrics[cat]
        print(f"  {cat:25s} F1: {m['f1']:.4f} (P: {m['precision']:.3f}, R: {m['recall']:.3f}, Sup: {m['support']})")

    if len(categories) > 10:
        print("\nBottom 5 categories by F1:")
        for cat in sorted_cats[-5:]:
            m = metrics[cat]
            print(f"  {cat:25s} F1: {m['f1']:.4f} (P: {m['precision']:.3f}, R: {m['recall']:.3f}, Sup: {m['support']})")


def main():
    """Main execution function."""
    args = parse_args()

    predictions_path = Path(args.predictions)

    if not args.quiet:
        print("="*80)
        print("CLASSIFICATION EVALUATION")
        print("="*80)
        print(f"Predictions: {predictions_path}")

    # Load predictions
    if not args.quiet:
        print("\nLoading predictions...")

    y_true, y_pred, categories, test_paths = load_predictions(predictions_path)

    if not args.quiet:
        print(f"  Samples: {len(y_true)}")
        print(f"  Categories: {len(categories)}")

    # Convert to dict format for metrics calculation
    y_true_dict = convert_to_dict_format(y_true, categories)
    y_pred_dict = convert_to_dict_format(y_pred, categories)

    # Calculate metrics
    if not args.quiet:
        print("\nCalculating metrics...")

    metrics = calculate_multilabel_metrics(y_true_dict, y_pred_dict)
    per_image_metrics = calculate_per_image_metrics(y_true_dict, y_pred_dict)

    # Print metrics
    if not args.quiet:
        print_metrics(metrics, per_image_metrics)

    # Save metrics to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            'timestamp': datetime.now().isoformat(),
            'predictions_file': str(predictions_path),
            'n_samples': len(y_true),
            'n_categories': len(categories),
            'categories': categories,
            'per_image_metrics': per_image_metrics,
            'metrics': metrics
        }

        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')

        if not args.quiet:
            print(f"\n✓ Metrics saved: {output_path}")

    if not args.quiet:
        print("\n" + "="*80)
        print("✓ Evaluation completed!")
        print("="*80)


if __name__ == "__main__":
    main()
