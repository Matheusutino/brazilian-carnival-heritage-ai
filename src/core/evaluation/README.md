# Evaluation Module

The `src/core/evaluation` module provides comprehensive evaluation capabilities for image classification, caption generation, and image quality assessment tasks.

## Overview

This module offers three distinct evaluation domains:

1. **Classification Metrics** (`classification_metrics.py`) - Binary and multilabel classification evaluation with micro/macro averaging
2. **Caption Metrics** (`caption_metrics.py`) - Text similarity metrics for evaluating generated captions (BLEU, METEOR, ROUGE, BERTScore)
3. **Image Quality Metrics** (`upscale_metrics.py`) - No-reference image quality assessment for upscaling evaluation

## Files

### `classification_metrics.py`

Metrics calculation for binary and multilabel classification tasks using scikit-learn.

**Key Features:**
- Binary classification metrics (precision, recall, F1, accuracy)
- Confusion matrix components (TP, FP, TN, FN)
- Multilabel metrics with micro/macro/weighted averaging
- Per-category detailed metrics
- Per-image multilabel metrics (exact match, Hamming loss, Jaccard score)

**Main Functions:**

#### `calculate_binary_metrics(y_true, y_pred)`
Calculate metrics for a single binary classification category.

**Returns:** Dictionary with TP, FP, TN, FN, precision, recall, F1, accuracy, support

**Usage Example:**
```python
from src.core.evaluation.classification_metrics import calculate_binary_metrics

y_true = [1, 0, 1, 1, 0, 1, 0]
y_pred = [1, 0, 0, 1, 0, 1, 1]

metrics = calculate_binary_metrics(y_true, y_pred)
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1: {metrics['f1']:.3f}")
print(f"TP: {metrics['tp']}, FP: {metrics['fp']}")
```

#### `calculate_multilabel_metrics(y_true_dict, y_pred_dict)`
Calculate metrics for multilabel classification across all categories.

**Args:**
- `y_true_dict`: Dictionary mapping category names to ground truth labels
- `y_pred_dict`: Dictionary mapping category names to predicted labels

**Returns:** Nested dictionary with per-category metrics and aggregates (micro_avg, macro_avg, weighted_avg)

**Usage Example:**
```python
from src.core.evaluation.classification_metrics import calculate_multilabel_metrics

y_true = {
    'carnival': [1, 0, 1, 1, 0],
    'street': [0, 1, 1, 0, 1],
    'costume': [1, 1, 0, 1, 0]
}

y_pred = {
    'carnival': [1, 0, 0, 1, 0],
    'street': [0, 1, 1, 1, 1],
    'costume': [1, 1, 1, 1, 0]
}

metrics = calculate_multilabel_metrics(y_true, y_pred)

# Per-category metrics
print(f"Carnival F1: {metrics['carnival']['f1']:.3f}")

# Aggregate metrics
print(f"Micro-avg F1: {metrics['micro_avg']['f1']:.3f}")
print(f"Macro-avg F1: {metrics['macro_avg']['f1']:.3f}")
print(f"Weighted-avg F1: {metrics['weighted_avg']['f1']:.3f}")
```

#### `calculate_per_image_metrics(y_true_dict, y_pred_dict)`
Calculate per-image metrics for multilabel classification.

**Returns:** Dictionary with exact_match_ratio, hamming_loss, jaccard_score

**Usage Example:**
```python
from src.core.evaluation.classification_metrics import calculate_per_image_metrics

metrics = calculate_per_image_metrics(y_true, y_pred)
print(f"Exact match ratio: {metrics['exact_match_ratio']:.2%}")
print(f"Hamming loss: {metrics['hamming_loss']:.3f}")
print(f"Jaccard score: {metrics['jaccard_score']:.3f}")
```

### `caption_metrics.py`

Text similarity metrics for evaluating generated captions against reference descriptions.

**Key Features:**
- BLEU scores (1-4 grams) with smoothing
- METEOR score (considers synonyms and stemming)
- ROUGE-L score (longest common subsequence)
- BERTScore (semantic similarity using Portuguese BERT)
- Batch processing with statistical aggregation
- Per-image detailed results

**Main Functions:**

#### `calculate_bleu_scores(reference, hypothesis)`
Calculate BLEU scores for a single text pair.

**Returns:** Dictionary with bleu_1, bleu_2, bleu_3, bleu_4 scores

#### `calculate_meteor(reference, hypothesis)`
Calculate METEOR score for a single text pair.

**Returns:** Float (0-1)

#### `calculate_rouge_l(reference, hypothesis)`
Calculate ROUGE-L F1 score for a single text pair.

**Returns:** Float (0-1)

#### `calculate_bertscore_batch(references, hypotheses, model='neuralmind/bert-base-portuguese-cased')`
Calculate BERTScore for multiple text pairs.

**Returns:** Tuple of (precision_list, recall_list, f1_list)

#### `evaluate_caption_pairs(references, hypotheses, filenames=None, bert_model=None, verbose=True)`
Comprehensive evaluation of caption pairs with all metrics.

**Returns:** Dictionary with aggregated metrics (mean, std, min, max, median) and per-image results

**Usage Example:**
```python
from src.core.evaluation.caption_metrics import evaluate_caption_pairs, print_metrics_summary

# Prepare data
references = [
    "Grupo de foliões fantasiados desfilando pela Avenida Rio Branco",
    "Bateria do bloco tocando instrumentos de percussão"
]

hypotheses = [
    "Pessoas fantasiadas em desfile de carnaval na avenida",
    "Músicos tocando tambores e instrumentos em apresentação"
]

filenames = ["photo_001.jpg", "photo_002.jpg"]

# Evaluate
results = evaluate_caption_pairs(
    references=references,
    hypotheses=hypotheses,
    filenames=filenames,
    bert_model='neuralmind/bert-base-portuguese-cased',
    verbose=True
)

# Print summary
print_metrics_summary(results, caption_type="LLM Captions")

# Access specific metrics
print(f"\nAverage BLEU-4: {results['metrics']['bleu_4']['mean']:.3f}")
print(f"Average BERTScore F1: {results['metrics']['bertscore_f1']['mean']:.3f}")

# Access per-image results
for item in results['per_image']:
    print(f"{item['filename']}: METEOR = {item['meteor']:.3f}")
```

### `upscale_metrics.py`

No-reference image quality assessment using fast OpenCV-based metrics.

**Key Features:**
- Sharpness measurement (Variance of Laplacian)
- Edge strength calculation (Sobel operators)
- Contrast assessment (standard deviation)
- Artifact detection (over-sharpening, ringing)
- Self-consistency check (PSNR with downsampled version)
- Batch processing with multiprocessing
- Method comparison utilities

**Class:** `ImageQualityEvaluator`

**Main Methods:**

#### `calculate_sharpness(img)`
Calculate image sharpness using Variance of Laplacian.

**Returns:** Float (higher = sharper)

#### `calculate_edge_strength(img)`
Calculate edge strength using Sobel operators.

**Returns:** Float (higher = more edges)

#### `calculate_contrast(img)`
Calculate image contrast.

**Returns:** Float (higher = more contrast)

#### `calculate_artifacts(img)`
Detect upscaling artifacts.

**Returns:** Dictionary with oversharp and ringing scores

#### `calculate_self_consistency(img_original, img_upscaled)`
Calculate PSNR between original and downsampled upscaled image.

**Returns:** Float (PSNR in dB, higher = better consistency)

#### `evaluate_image(image_path)`
Evaluate all quality metrics for a single image.

**Returns:** Dictionary with all quality scores

#### `evaluate_batch(image_paths, verbose=True, n_workers=4)`
Evaluate multiple images in parallel.

**Returns:** List of dictionaries with quality scores

#### `compare_methods(base_dir, verbose=True, n_workers=4)`
Compare quality across different upscaling methods.

**Returns:** Dictionary with aggregate statistics per method

**Usage Example:**
```python
from pathlib import Path
from src.core.evaluation.upscale_metrics import ImageQualityEvaluator

# Initialize evaluator
evaluator = ImageQualityEvaluator(
    original_images_dir='dataset/downloaded'
)

# Evaluate single image
result = evaluator.evaluate_image('upscaled/photo_001_realesrgan_4x.jpg')
print(f"Sharpness: {result['sharpness']:.2f}")
print(f"Edge strength: {result['edge_strength']:.2f}")
print(f"Self-consistency (PSNR): {result['self_consistency']:.2f} dB")

# Evaluate batch of images
image_paths = list(Path('upscaled/').glob('*.jpg'))
results = evaluator.evaluate_batch(image_paths, verbose=True, n_workers=4)

# Calculate averages
avg_sharpness = sum(r['sharpness'] for r in results) / len(results)
print(f"Average sharpness: {avg_sharpness:.2f}")

# Compare multiple upscaling methods
comparison = evaluator.compare_methods(
    base_dir='results/upscaling/',
    verbose=True,
    n_workers=4
)

for method_name, stats in comparison.items():
    print(f"\n{method_name}:")
    print(f"  Sharpness: {stats['sharpness_mean']:.2f} ± {stats['sharpness_std']:.2f}")
    print(f"  Self-consistency: {stats['self_consistency_mean']:.2f} dB")
```

### `__init__.py`

Package initialization with module exports.

## Metrics Overview

### Classification Metrics

**Per-Category Metrics:**
- **Precision**: TP / (TP + FP) - Fraction of positive predictions that are correct
- **Recall**: TP / (TP + FN) - Fraction of actual positives that are detected
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: (TP + TN) / Total - Fraction of correct predictions
- **Support**: Number of actual positive samples

**Aggregate Metrics:**
- **Micro-average**: Aggregate all labels, then calculate (weighted by frequency)
- **Macro-average**: Calculate per-label, then average (equal weight per label)
- **Weighted-average**: Calculate per-label, then weighted average by support

**Per-Image Metrics:**
- **Exact Match Ratio**: Fraction of images with all labels correct
- **Hamming Loss**: Average fraction of wrong labels per image
- **Jaccard Score**: Average intersection-over-union across images

### Caption Metrics

**N-gram Overlap Metrics:**
- **BLEU-1 to BLEU-4**: Precision-based n-gram matching (1 to 4-grams)
- **METEOR**: Considers synonyms, stemming, word order
- **ROUGE-L**: Longest common subsequence F1

**Semantic Similarity:**
- **BERTScore**: Contextual embeddings similarity using Portuguese BERT
  - Precision: How much of hypothesis is in reference
  - Recall: How much of reference is in hypothesis
  - F1: Harmonic mean of precision and recall

### Image Quality Metrics

**Sharpness & Detail:**
- **Sharpness**: Variance of Laplacian (measures focus)
- **Edge Strength**: Mean Sobel magnitude (measures detail preservation)
- **Contrast**: Standard deviation of grayscale (measures dynamic range)

**Artifact Detection:**
- **Over-sharpening**: 99th percentile of Laplacian (detects excessive sharpening)
- **Ringing**: Standard deviation of Sobel (detects edge halos)

**Consistency:**
- **Self-Consistency (PSNR)**: Quality when downsampled back to original resolution

## Output Formats

### Binary Classification Metrics
```python
{
    'tp': 15,           # True positives
    'fp': 3,            # False positives
    'tn': 82,           # True negatives
    'fn': 5,            # False negatives
    'precision': 0.833,
    'recall': 0.750,
    'f1': 0.789,
    'accuracy': 0.924,
    'support': 20       # Total positive samples
}
```

### Multilabel Classification Metrics
```python
{
    'carnival': {
        'tp': 45, 'fp': 8, 'tn': 120, 'fn': 7,
        'precision': 0.849, 'recall': 0.865, 'f1': 0.857,
        'accuracy': 0.917, 'support': 52
    },
    # ... more categories
    'micro_avg': {
        'precision': 0.835, 'recall': 0.821, 'f1': 0.828,
        'accuracy': 0.912, 'total_support': 450
    },
    'macro_avg': {
        'precision': 0.798, 'recall': 0.784, 'f1': 0.791,
        'accuracy': 0.895, 'total_support': 450
    },
    'weighted_avg': {
        'precision': 0.823, 'recall': 0.809, 'f1': 0.816,
        'accuracy': 0.907, 'total_support': 450
    }
}
```

### Caption Evaluation Results
```python
{
    'num_pairs': 50,
    'metrics': {
        'bleu_1': {'mean': 0.654, 'std': 0.123, 'min': 0.321, 'max': 0.892, 'median': 0.671},
        'bleu_4': {'mean': 0.234, 'std': 0.089, 'min': 0.045, 'max': 0.456, 'median': 0.245},
        'meteor': {'mean': 0.456, 'std': 0.102, 'min': 0.201, 'max': 0.678, 'median': 0.467},
        'rouge_l': {'mean': 0.543, 'std': 0.114, 'min': 0.234, 'max': 0.789, 'median': 0.556},
        'bertscore_f1': {'mean': 0.823, 'std': 0.067, 'min': 0.654, 'max': 0.934, 'median': 0.834}
    },
    'per_image': [
        {
            'filename': 'photo_001.jpg',
            'reference': 'Reference text...',
            'hypothesis': 'Generated text...',
            'bleu_1': 0.678, 'bleu_2': 0.534, 'bleu_3': 0.389, 'bleu_4': 0.267,
            'meteor': 0.489, 'rouge_l': 0.567,
            'bertscore_precision': 0.834, 'bertscore_recall': 0.812, 'bertscore_f1': 0.823
        },
        # ... more images
    ]
}
```

### Image Quality Results
```python
{
    'image_path': '/path/to/image_realesrgan_4x.jpg',
    'image_name': 'image_realesrgan_4x.jpg',
    'status': 'success',
    'sharpness': 234.56,
    'edge_strength': 12.34,
    'contrast': 45.67,
    'oversharp': 156.78,
    'ringing': 23.45,
    'self_consistency': 32.1  # PSNR in dB
}
```

## Performance Considerations

### Classification Metrics
- Very fast computation using NumPy and scikit-learn
- Suitable for large-scale evaluation (millions of predictions)
- Per-image metrics slightly slower than aggregate metrics

### Caption Metrics
- **BLEU/METEOR/ROUGE**: Fast, suitable for thousands of pairs
- **BERTScore**: Slower, GPU recommended for large batches
  - Uses transformer models (loads once, reused for batch)
  - Typical: ~10-50 pairs/second on GPU, ~1-5 pairs/second on CPU

### Image Quality Metrics
- **Single Image**: ~50-200ms per image (CPU)
- **Batch Processing**: Uses multiprocessing for parallelization
  - Adjust `n_workers` based on CPU cores
  - Typical: 4-8 workers for 4-8 core systems
- **Self-Consistency**: Requires original images, adds ~20-30ms per image

## Error Handling

### Classification Metrics
- Handles missing categories in predictions (treats as zeros)
- Zero division protection for edge cases
- Validates input shapes and types

### Caption Metrics
- Automatic NLTK data download on first use
- Fallback to default BERT model if specified model unavailable
- Handles empty or mismatched input lists

### Image Quality Metrics
- Returns error status for failed image loads
- Handles missing original images (self-consistency = None)
- Multiprocessing exceptions caught and reported per image

## Integration with Project

This module supports evaluation across the entire Bloco Suvaco de Cristo pipeline:

### Classification Evaluation
```python
# Load predictions
import numpy as np
data = np.load('results/predictions.npz')
y_true = data['y_true']
y_pred = data['y_pred']
categories = data['categories']

# Convert to dict format
from src.core.evaluation.classification_metrics import calculate_multilabel_metrics

y_true_dict = {cat: y_true[:, i].tolist() for i, cat in enumerate(categories)}
y_pred_dict = {cat: y_pred[:, i].tolist() for i, cat in enumerate(categories)}

# Calculate metrics
metrics = calculate_multilabel_metrics(y_true_dict, y_pred_dict)
```

### Caption Evaluation
```python
import json
from src.core.evaluation.caption_metrics import evaluate_caption_pairs

# Load caption results
with open('results/captions.json') as f:
    caption_data = json.load(f)

references = [item['reference'] for item in caption_data]
hypotheses = [item['generated'] for item in caption_data]

# Evaluate
results = evaluate_caption_pairs(references, hypotheses)
```

### Image Quality Evaluation
```python
from pathlib import Path
from src.core.evaluation.upscale_metrics import ImageQualityEvaluator

# Compare upscaling methods
evaluator = ImageQualityEvaluator()
comparison = evaluator.compare_methods('results/upscaling/')

# Find best method
best_method = max(comparison.items(),
                 key=lambda x: x[1]['sharpness_mean'])
print(f"Best method: {best_method[0]}")
```
