# Classification Module

The `src/core/classification` module provides image classification capabilities using two complementary approaches: classical machine learning models with feature embeddings and multimodal Large Language Models (LLMs).

## Overview

This module offers two distinct classification strategies:

1. **Classical ML Classifiers** (`multilabel_classifiers.py`) - Fast, efficient multilabel classification using scikit-learn and gradient boosting models with Bayesian hyperparameter optimization
2. **Multimodal LLM Classification** (`llm_classifier.py`) - Zero-shot image classification using vision-language models

Both approaches support **multilabel classification**, where each image can be assigned multiple categories simultaneously.

## Files

### `multilabel_classifiers.py`

Multilabel classification pipeline using classical ML models with automated hyperparameter optimization via Optuna.

**Key Features:**
- Support for 7 classifier types:
  - Linear Discriminant Analysis (LDA)
  - Random Forest (RF)
  - XGBoost (XGB)
  - LightGBM (LGBM)
  - Naive Bayes (NB)
  - K-Nearest Neighbors (KNN)
  - Decision Tree (Tree)
- Bayesian hyperparameter optimization with Optuna
- Iterative stratification for balanced cross-validation in multilabel settings
- Comprehensive multilabel metrics (micro/macro averaging, per-category metrics)
- Model persistence (save/load trained models)
- GPU acceleration support (XGBoost, LightGBM)

**Classes:**
- `MultilabelClassifierFactory` - Factory for creating classifiers with predefined hyperparameter spaces
- `MultilabelClassifierPipeline` - End-to-end pipeline for training, optimization, evaluation, and deployment

**Usage Example:**
```python
from pathlib import Path
import numpy as np
from src.core.classification.multilabel_classifiers import MultilabelClassifierPipeline

# Prepare data (features from embeddings, binary labels)
X_train = np.load('train_embeddings.npy')  # Shape: (n_samples, n_features)
y_train = np.load('train_labels.npy')      # Shape: (n_samples, n_labels)
X_test = np.load('test_embeddings.npy')
y_test = np.load('test_labels.npy')

categories = ['carnival', 'street', 'costume', 'crowd', 'music']

# Initialize pipeline with XGBoost
pipeline = MultilabelClassifierPipeline(
    classifier_type='xgb',
    categories=categories
)

# Optimize hyperparameters and train
pipeline.optimize_and_train(
    X_train, y_train,
    n_trials=50,           # Number of Optuna trials
    cv=3,                  # Cross-validation folds
    optimize_for='f1_macro',  # Optimization metric
    verbose=True
)

# Evaluate on test set
metrics = pipeline.evaluate(X_test, y_test, verbose=True)
print(f"Test F1 (macro): {metrics['f1_macro']:.4f}")
print(f"Exact match accuracy: {metrics['exact_match_accuracy']:.4f}")

# Save trained model
pipeline.save('models/xgb_classifier.pkl')

# Load and use later
loaded_pipeline = MultilabelClassifierPipeline.load('models/xgb_classifier.pkl')
predictions = loaded_pipeline.predict(X_test)
```

**Training Without Optimization:**
```python
# Train with default or custom parameters (faster, no optimization)
pipeline = MultilabelClassifierPipeline(classifier_type='rf', categories=categories)
pipeline.train(
    X_train, y_train,
    params={'n_estimators': 100, 'max_depth': 20}
)
```

### `llm_classifier.py`

Image classifier using multimodal LLMs for zero-shot multilabel classification.

**Key Features:**
- Zero-shot classification (no training data required)
- JSON schema validation for output consistency
- Automatic retry mechanism with temperature randomization
- Configurable prompts via YAML configuration
- Base64 image encoding with automatic resizing
- Parallel batch processing with thread pool
- Support for arbitrary category lists

**Class:** `ImageClassifier`

**Usage Example:**
```python
from pathlib import Path
from src.core.classification.llm_classifier import ImageClassifier, load_categories

# Load categories from CSV file
categories = load_categories('data/categories.csv')
# Or define manually
categories = ['carnival', 'street_party', 'costume', 'crowd', 'musical_instruments']

# Initialize classifier
classifier = ImageClassifier(
    config_path='config/prompts.yaml',
    categories=categories,
    provider='nvidia',
    model_name='meta/llama-3.2-90b-vision-instruct',
    temperature=0.7,
    max_tokens=500,
    max_retries=3,
    prompt_key='classification/zero_shot',
    max_image_dim=768  # Resize large images
)

# Classify a single image
image_path = Path('carnival_photo.jpg')
result = classifier.classify(image_path)

if result['status'] == 'success':
    classification = result['classification']
    print("Classification results:")
    for category, value in classification.items():
        if value == 1:
            print(f"  ✓ {category}")

# Classify multiple images in parallel
image_paths = [Path(f'photo_{i}.jpg') for i in range(100)]
results = classifier.classify_batch(image_paths, max_workers=4)

# Extract predictions for analysis
successful_results = [r for r in results if r['status'] == 'success']
```

### `__init__.py`

Package initialization file.

## Architecture

### Classical ML Pipeline

```
Image Embeddings → Feature Vector → ML Classifier → Binary Labels per Category
                                          ↓
                                  Hyperparameter Optimization
                                  (Optuna + Stratified CV)
```

### LLM Classification Pipeline

```
Image → Base64 Encoding → Prompt Construction → LLM API → JSON Response → Schema Validation → Binary Labels
```

## Configuration

### MultilabelClassifierPipeline Configuration

**Available Classifier Types:**

| Type | Name | Best For | GPU Support |
|------|------|----------|-------------|
| `lda` | Linear Discriminant Analysis | Linear separability, low-dimensional | No |
| `rf` | Random Forest | General purpose, interpretability | No |
| `xgb` | XGBoost | High performance, large datasets | Yes |
| `lgbm` | LightGBM | Fast training, large datasets | Yes |
| `nb` | Naive Bayes | Baseline, probabilistic | No |
| `knn` | K-Nearest Neighbors | Small datasets, similarity-based | No |
| `tree` | Decision Tree | Interpretability, baseline | No |

**Hyperparameter Spaces:**

Each classifier has a predefined hyperparameter search space optimized by Optuna's Tree-structured Parzen Estimator (TPE) sampler:

```python
# Example: XGBoost hyperparameter space
{
    'n_estimators': (50, 300),        # Integer range
    'max_depth': (3, 10),             # Integer range
    'learning_rate': (0.01, 0.3),     # Log-uniform distribution
    'subsample': (0.6, 1.0),          # Uniform distribution
    'colsample_bytree': (0.6, 1.0)    # Uniform distribution
}
```

### ImageClassifier Configuration

```python
ImageClassifier(
    config_path='config/prompts.yaml',           # YAML configuration file
    categories=['cat1', 'cat2', ...],            # List of category names
    provider='nvidia',                            # LLM provider
    model_name='meta/llama-3.2-90b-vision-instruct',
    temperature=0.7,                              # Generation temperature
    max_tokens=500,                               # Maximum output tokens
    max_retries=3,                                # Retry attempts on failure
    retry_temp_min=0.0,                           # Min temperature for retries
    retry_temp_max=1.0,                           # Max temperature for retries
    prompt_key='classification/zero_shot',        # Key in YAML config
    max_image_dim=768                             # Optional image resizing
)
```

### YAML Configuration Structure

The ImageClassifier requires a YAML configuration file:

```yaml
classification:
  zero_shot:
    system_prompt: |
      You are an expert image classifier for carnival and cultural heritage photographs.

    user_prompt: |
      Classify this image for the following categories. Return 1 if present, 0 if not.

      Categories:
      {categories_list}

      Image:
      {image_tag}

      Return ONLY a JSON object with category names as keys and 0/1 as values.

    output_schema:
      type: object
      additionalProperties:
        type: integer
        enum: [0, 1]
```

## Output Formats

### MultilabelClassifierPipeline Output

**Prediction Output:**
```python
predictions = pipeline.predict(X_test)
# Shape: (n_samples, n_labels)
# Values: 0 or 1 for each category
array([[1, 0, 1, 0, 1],
       [0, 1, 0, 1, 0],
       ...])
```

**Evaluation Metrics:**
```python
{
    'exact_match_accuracy': 0.6543,  # All labels correct
    'accuracy_micro': 0.8234,        # Label-level accuracy
    'precision_micro': 0.8156,       # Overall precision
    'recall_micro': 0.7892,          # Overall recall
    'f1_micro': 0.8021,              # Overall F1
    'precision_macro': 0.7945,       # Average precision across categories
    'recall_macro': 0.7623,          # Average recall across categories
    'f1_macro': 0.7781,              # Average F1 across categories
    'per_category': {
        'carnival': {
            'accuracy': 0.92,
            'precision': 0.88,
            'recall': 0.85,
            'f1': 0.865,
            'support': 150
        },
        # ... more categories
    }
}
```

### ImageClassifier Output

```python
{
    'image_path': '/path/to/image.jpg',
    'image_name': 'image.jpg',
    'timestamp': '2025-10-24T10:30:00',
    'status': 'success',
    'metadata': {},
    'attempts': [
        {
            'attempt': 1,
            'temperature': 0.7,
            'success': True,
            'error': None
        }
    ],
    'classification': {
        'carnival': 1,
        'street_party': 1,
        'costume': 1,
        'crowd': 1,
        'musical_instruments': 0
    },
    'temperature': 0.7,
    'raw_response': '{"carnival": 1, "street_party": 1, ...}'
}
```

## Evaluation Metrics

### Multilabel Classification Metrics

The module provides comprehensive multilabel metrics:

**Exact Match Accuracy (Subset Accuracy):**
- Fraction of samples where ALL labels are predicted correctly
- Strictest metric, most challenging for multilabel problems

**Micro-averaged Metrics:**
- Aggregates contributions across all labels
- Weighted by label frequency
- Better for imbalanced datasets

**Macro-averaged Metrics:**
- Unweighted average across all labels
- Treats all categories equally
- Better for balanced evaluation

**Per-category Metrics:**
- Individual precision, recall, F1, and support for each category
- Identifies strong/weak categories

## Performance Considerations

### MultilabelClassifierPipeline

**Optimization Speed:**
- LightGBM: Fastest training (GPU support)
- XGBoost: Best performance-speed tradeoff (GPU support)
- Random Forest: Moderate speed, good baseline
- KNN/Tree: Fast optimization, slower prediction

**Recommendations:**
- Use `optimize_for='f1_macro'` for balanced datasets
- Use `optimize_for='f1_micro'` for imbalanced datasets
- Start with `n_trials=50` for initial exploration
- Increase to `n_trials=100-200` for production models
- Use `cv=3` for faster optimization, `cv=5` for more robust estimates

### ImageClassifier

**Parallel Processing:**
- Adjust `max_workers` based on API rate limits (typical: 2-8)
- Higher parallelism increases throughput but may hit rate limits

**Retry Strategy:**
- `max_retries=3` balances reliability and speed
- Temperature randomization helps overcome LLM inconsistencies
- Failed attempts are logged for debugging

**Image Resizing:**
- Use `max_image_dim=768` to reduce API payload and costs
- Smaller dimensions (512-768) usually sufficient for classification
- Larger dimensions (1024+) for fine-grained visual details

## Error Handling

### MultilabelClassifierPipeline

- Handles warnings from sklearn (e.g., single-sample classes)
- Graceful degradation during cross-validation (assigns 0.0 to failed folds)
- Validation before save/load operations

### ImageClassifier

- Image loading errors are caught and reported in results
- Network and API errors trigger retry mechanisms
- Batch processing continues even if individual images fail
- Status field indicates 'success' or 'error' for each image
- Detailed attempt history for debugging

## Integration with Project

This module is specifically designed for the **Bloco Suvaco de Cristo** historical archive project, with capabilities for:

- **Multilabel Cultural Tagging**: Assign multiple relevant categories to carnival photographs
- **Zero-shot Classification**: Classify new image types without retraining
- **Scalable Processing**: Batch processing for large historical archives
- **Interpretability**: Per-category metrics help understand model behavior

## Comparison: Classical ML vs. LLM Classification

| Aspect | Classical ML | LLM |
|--------|--------------|-----|
| **Training** | Requires labeled data + feature extraction | Zero-shot (no training) |
| **Speed** | Very fast inference | Slower (API calls) |
| **Accuracy** | High (with good features) | Variable (depends on prompt) |
| **Cost** | Low (one-time training) | Per-image API cost |
| **Flexibility** | Fixed categories | Easy to add new categories |
| **Interpretability** | High (feature importance) | Lower (black box) |
| **Best For** | Production systems, large-scale | Prototyping, flexible needs |
