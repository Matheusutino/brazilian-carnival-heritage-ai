# Data Module

The `src/core/data` module provides dataset loading and management utilities for multilabel classification tasks.

## Overview

This module handles loading train/test splits, managing image paths, and providing access to multilabel annotation data in a consistent format.

## Files

### `dataset_loader.py`

Dataset container for multilabel classification with train/test split management.

**Key Features:**
- Loads train/test splits from JSON configuration files
- Automatically filters valid image files by extension
- Extracts multilabel annotations from CSV files
- Provides convenient access to data splits and statistics
- Label distribution analysis
- Consistent data format for classification pipelines

**Class:** `MultilabelDataset`

**Usage Example:**
```python
from pathlib import Path
from src.core.data.dataset_loader import MultilabelDataset

# Load dataset from split file
dataset = MultilabelDataset(
    split_file='data/splits/train_test_split.json'
)

# Get training data
train_paths, y_train = dataset.get_train_data()
print(f"Training samples: {len(train_paths)}")
print(f"Label matrix shape: {y_train.shape}")  # (n_samples, n_labels)

# Get test data
test_paths, y_test = dataset.get_test_data()
print(f"Test samples: {len(test_paths)}")

# Get category names
categories = dataset.get_categories()
print(f"Categories: {categories}")

# Get sample counts
counts = dataset.get_n_samples()
print(f"Train: {counts['train']}, Test: {counts['test']}")

# Analyze label distribution
distribution = dataset.get_label_distribution()
print("\nTrain label distribution:")
for category, count in distribution['train'].items():
    print(f"  {category}: {count}")

print("\nTest label distribution:")
for category, count in distribution['test'].items():
    print(f"  {category}: {count}")
```

### `__init__.py`

Package initialization file.

## Split File Format

The dataset loader expects a JSON split file with the following structure:

```json
{
  "dataset": "path/to/main_dataset.csv",
  "categories": "path/to/categories_labels.csv",
  "categories_list": ["category1", "category2", "category3", ...],
  "train_indices": [0, 1, 5, 8, 10, ...],
  "test_indices": [2, 3, 4, 6, 7, ...],
  "split_date": "2025-10-24T10:30:00",
  "train_size": 800,
  "test_size": 200,
  "split_ratio": 0.8
}
```

**Key Fields:**
- `dataset`: Path to main dataset CSV with `LocalPath` and `Item` columns
- `categories`: Path to labels CSV with binary annotations per category
- `categories_list`: Ordered list of category names (column names in labels CSV)
- `train_indices`: Row indices for training set
- `test_indices`: Row indices for test set

## Dataset CSV Format

### Main Dataset CSV

Required columns:
- `Item`: Unique identifier for each item (filename or ID)
- `LocalPath`: Path to image file

Example:
```csv
Item,LocalPath,CreationDate,FileSize
photo_001.jpg,/path/to/photo_001.jpg,2024-01-15,2048576
photo_002.jpg,/path/to/photo_002.jpg,2024-02-20,1536000
```

### Categories/Labels CSV

Required columns:
- `Item`: Unique identifier matching main dataset
- One binary column (0 or 1) per category

Example:
```csv
Item,carnival,street,costume,crowd,music
photo_001.jpg,1,1,1,0,1
photo_002.jpg,0,1,0,1,0
photo_003.jpg,1,0,1,1,1
```

## Supported Image Formats

The loader automatically filters for valid image file extensions:
- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- GIF (`.gif`)
- BMP (`.bmp`)
- WebP (`.webp`)
- TIFF (`.tiff`, `.tif`)

Files with other extensions are automatically excluded from the dataset.

## Data Output Format

### Image Paths
```python
train_paths = [
    '/path/to/photo_001.jpg',
    '/path/to/photo_005.jpg',
    ...
]
```

### Label Matrices
```python
# y_train shape: (n_samples, n_labels)
# Binary values: 0 or 1
y_train = array([[1, 0, 1, 0, 1],
                 [0, 1, 0, 1, 0],
                 [1, 1, 1, 1, 0],
                 ...])
```

### Label Distribution
```python
{
    'train': {
        'carnival': 450,
        'street': 320,
        'costume': 280,
        'crowd': 510,
        'music': 190
    },
    'test': {
        'carnival': 112,
        'street': 85,
        'costume': 68,
        'crowd': 125,
        'music': 48
    }
}
```

## Integration with Classification Pipeline

The dataset loader is designed to work seamlessly with the classification module:

```python
from src.core.data.dataset_loader import MultilabelDataset
from src.core.classification.multilabel_classifiers import MultilabelClassifierPipeline
import numpy as np

# Load dataset
dataset = MultilabelDataset('data/splits/train_test_split.json')
train_paths, y_train = dataset.get_train_data()
test_paths, y_test = dataset.get_test_data()

# Load pre-computed embeddings (from embedding module)
X_train = np.load('embeddings/train_embeddings.npy')
X_test = np.load('embeddings/test_embeddings.npy')

# Train classifier
pipeline = MultilabelClassifierPipeline(
    classifier_type='xgb',
    categories=dataset.get_categories()
)
pipeline.optimize_and_train(X_train, y_train, n_trials=50)

# Evaluate
metrics = pipeline.evaluate(X_test, y_test)
```

## Data Validation

The loader includes several validation steps:

1. **File Existence**: Verifies split file and referenced CSV files exist
2. **Image Filtering**: Excludes non-image files and missing paths
3. **Data Merging**: Joins main dataset with labels on `Item` column
4. **Missing Values**: Fills missing labels with 0 (assumes absent label = negative)
5. **Type Conversion**: Ensures labels are integer type (0 or 1)

## Error Handling

The loader provides clear error messages for common issues:

```python
try:
    dataset = MultilabelDataset('invalid_path.json')
except FileNotFoundError as e:
    print(f"Split file not found: {e}")

try:
    dataset = MultilabelDataset('split.json')
    train_paths, y_train = dataset.get_train_data()
except KeyError as e:
    print(f"Missing required column: {e}")
```

## Methods Reference

### `__init__(split_file, labels_source=None)`
Initialize dataset from split file.

**Args:**
- `split_file`: Path to JSON split configuration
- `labels_source` (optional): Alternative labels source (usually None)

### `get_train_data()`
Returns tuple of `(train_paths, y_train)`

### `get_test_data()`
Returns tuple of `(test_paths, y_test)`

### `get_categories()`
Returns list of category names

### `get_n_samples()`
Returns dict with `'train'` and `'test'` sample counts

### `get_label_distribution()`
Returns dict with label counts per category for train and test sets

## Integration with Project

This module is part of the **Bloco Suvaco de Cristo** archive classification system:

- **Stratified Splits**: Works with train/test splits created by iterative stratification for balanced multilabel distribution
- **Flexible Labels**: Supports arbitrary category sets for cultural tagging
- **Reproducibility**: Loads fixed splits for consistent evaluation across experiments
- **Metadata Preservation**: Maintains split metadata for experiment tracking
