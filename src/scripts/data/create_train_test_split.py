"""Create stratified train/test split for multilabel classification.

This script creates a 70/30 train/test split of the dataset using iterative
stratification to maintain label distribution. The split indices are saved
for reproducibility.
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Create stratified train/test split for multilabel classification'
    )

    parser.add_argument('--dataset', default='dataset/dataset_completo.csv',
                       help='Dataset CSV file')
    parser.add_argument('--categories', default='dataset/palavras_chave_processed.csv',
                       help='Categories CSV file')
    parser.add_argument('--output', default='dataset/splits/split_latest.json',
                       help='Output split file path')
    parser.add_argument('--test-size', type=float, default=0.3,
                       help='Test set proportion (default: 0.3)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility (always 42 for consistency)')

    return parser.parse_args()


def load_categories(categories_path):
    """Load categories from CSV file."""
    df_cat = pd.read_csv(categories_path)
    # Extract category names (all columns except Id and Item)
    categories = [col for col in df_cat.columns if col not in ['Id', 'Item']]
    return categories


def main():
    """Main execution function."""
    args = parse_args()

    print("="*80)
    print("MULTILABEL TRAIN/TEST SPLIT")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Test size: {args.test_size}")
    print(f"Random state: {args.random_state}")

    # Load categories
    categories = load_categories(args.categories)
    print(f"\nCategories: {len(categories)}")

    # Load dataset
    df = pd.read_csv(args.dataset)
    df = df[df['LocalPath'].notna()]

    # Filter only image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
    df = df[df['LocalPath'].str.lower().str.endswith(tuple(image_extensions))]

    # Filter out macOS metadata files (._*) and hidden files
    df = df[~df['LocalPath'].str.contains(r'/\._|/\.', regex=True, na=False)]
    df = df[~df['LocalPath'].apply(lambda x: Path(x).name.startswith('._') or Path(x).name.startswith('.'))]

    print(f"\nTotal images: {len(df)}")

    # Load labels from categories CSV
    print("\nLoading labels from categories CSV...")
    df_labels = pd.read_csv(args.categories)

    # Remove duplicates from labels (keep first occurrence per Item)
    df_labels = df_labels.drop_duplicates(subset='Item', keep='first')

    # Merge with main dataset on Item name
    df = df.merge(df_labels, left_on='Item', right_on='Item', how='inner', suffixes=('', '_label'))

    # Remove any remaining duplicates based on LocalPath (should not happen, but just in case)
    df = df.drop_duplicates(subset='LocalPath', keep='first')

    print(f"Images with labels: {len(df)}")

    # Extract label matrix and fill NaN with 0
    y = df[categories].fillna(0).values.astype(int)

    print(f"\nLabel matrix shape: {y.shape}")
    print(f"Images with at least one label: {(y.sum(axis=1) > 0).sum()}")
    print(f"Images with no labels: {(y.sum(axis=1) == 0).sum()}")

    print(f"\nLabel matrix shape: {y.shape}")
    print(f"Average labels per image: {y.sum(axis=1).mean():.2f}")
    print(f"Label distribution:")
    for i, cat in enumerate(categories):
        count = y[:, i].sum()
        print(f"  {cat}: {count} ({count/len(y)*100:.1f}%)")

    # Create feature matrix (just indices for now)
    X = np.arange(len(df)).reshape(-1, 1)

    # Perform iterative stratified split
    print(f"\nPerforming iterative stratified split...")
    X_train, y_train, X_test, y_test = iterative_train_test_split(
        X, y, test_size=args.test_size
    )

    train_indices = X_train.flatten().tolist()
    test_indices = X_test.flatten().tolist()

    print(f"\nTrain set: {len(train_indices)} images ({len(train_indices)/len(df)*100:.1f}%)")
    print(f"Test set: {len(test_indices)} images ({len(test_indices)/len(df)*100:.1f}%)")

    # Verify label distribution preservation
    print(f"\nLabel distribution in train set:")
    for i, cat in enumerate(categories):
        count = y_train[:, i].sum()
        print(f"  {cat}: {count} ({count/len(y_train)*100:.1f}%)")

    print(f"\nLabel distribution in test set:")
    for i, cat in enumerate(categories):
        count = y_test[:, i].sum()
        print(f"  {cat}: {count} ({count/len(y_test)*100:.1f}%)")

    # Create output directory
    output_file = Path(args.output)
    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save split metadata
    split_info = {
        'timestamp': datetime.now().isoformat(),
        'dataset': args.dataset,
        'categories': args.categories,
        'n_total': len(df),
        'n_train': len(train_indices),
        'n_test': len(test_indices),
        'test_size': args.test_size,
        'random_state': args.random_state,
        'categories_list': categories,
        'train_indices': train_indices,
        'test_indices': test_indices
    }

    output_file.write_text(json.dumps(split_info, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n✓ Split saved: {output_file}")

    # Save train/test DataFrames
    df_train = df.iloc[train_indices].copy()
    df_test = df.iloc[test_indices].copy()

    train_csv = output_dir / "train.csv"
    test_csv = output_dir / "test.csv"

    df_train.to_csv(train_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    print(f"✓ Train CSV: {train_csv}")
    print(f"✓ Test CSV: {test_csv}")

    print("\n" + "="*80)
    print("Split creation completed!")
    print("="*80)


if __name__ == "__main__":
    main()
