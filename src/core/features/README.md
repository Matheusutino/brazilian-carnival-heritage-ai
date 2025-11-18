# Features Module

The `src/core/features` module provides feature extraction capabilities using state-of-the-art pre-trained vision models. It converts images into dense vector representations suitable for downstream classification tasks.

## Overview

This module offers a unified interface for extracting visual features from images using multiple vision model architectures:

1. **DINOv2** - Self-supervised vision transformer from Meta AI
2. **ConvNeXtV2** - Modern convolutional architecture with masked autoencoding

Features are extracted as fixed-length vectors that capture semantic and visual information from images, enabling efficient classification without fine-tuning large models.

## Files

### `feature_extractor.py`

Feature extraction module with support for multiple vision model backends.

**Key Features:**
- Unified interface for DINOv2 and ConvNeXtV2 models
- Single image and batch processing
- Automatic GPU/CPU device selection
- Progress tracking for large datasets
- Feature persistence (save/load)
- Error handling with graceful degradation

**Classes:**

#### `FeatureExtractor`

Low-level feature extractor for individual images or small batches.

**Supported Models:**
- `dinov2-small` - DINOv2 Small (384-dim features)
- `dinov2-base` - DINOv2 Base (768-dim features)
- `convnextv2-tiny` - ConvNeXtV2 Tiny (768-dim features)
- `convnextv2-base` - ConvNeXtV2 Base (1024-dim features)

**Usage Example:**
```python
from pathlib import Path
from src.core.features.feature_extractor import FeatureExtractor
import numpy as np

# Initialize extractor
extractor = FeatureExtractor(
    model_name='dinov2-small',
    device='cuda'  # or 'cpu', or None for auto
)

print(f"Model: {extractor.model_name}")
print(f"Feature dimension: {extractor.feature_dim}")
print(f"Device: {extractor.device_str}")

# Extract from single image
image_path = Path('photo.jpg')
features = extractor.extract_from_path(image_path)
print(f"Features shape: {features.shape}")  # (384,) for dinov2-small

# Extract from PIL Image
from PIL import Image
image = Image.open('photo.jpg').convert('RGB')
features = extractor.extract_from_image(image)

# Extract from batch of images
image_paths = [Path(f'photo_{i}.jpg') for i in range(10)]
batch_features = extractor.extract_batch(image_paths)
print(f"Batch features shape: {batch_features.shape}")  # (10, 384)
```

#### `FeatureExtractionPipeline`

High-level pipeline for dataset-scale feature extraction with progress tracking and persistence.

**Usage Example:**
```python
from pathlib import Path
from src.core.features.feature_extractor import FeatureExtractionPipeline

# Initialize pipeline
pipeline = FeatureExtractionPipeline(
    model_name='dinov2-small',
    device='cuda'
)

# Extract features from dataset
image_paths = [Path(f'dataset/{i}.jpg') for i in range(1000)]

# Option 1: Extract and save
features = pipeline.extract_and_save(
    image_paths=image_paths,
    output_file='features/dinov2_small_features.npz',
    batch_size=32,
    metadata={'dataset': 'bloco_suvaco', 'split': 'train'}
)

# Option 2: Extract only
features = pipeline.extract_from_paths(
    image_paths=image_paths,
    batch_size=32,
    show_progress=True
)

# Load saved features
data = FeatureExtractionPipeline.load_features('features/dinov2_small_features.npz')
features = data['features']
paths = data['image_paths']
model_name = str(data['model_name'])
feature_dim = int(data['feature_dim'])

print(f"Loaded {features.shape[0]} features of dimension {feature_dim}")
```

### `__init__.py`

Package initialization file.

## Model Architectures

### DINOv2 (Self-Supervised Vision Transformer)

**Description:**
- Self-supervised learning via distillation
- Vision Transformer (ViT) architecture
- Trained on diverse image datasets
- Excellent for transfer learning without labels

**Variants:**
- **dinov2-small**: 384-dim features, ~22M parameters
- **dinov2-base**: 768-dim features, ~86M parameters

**Best For:**
- General-purpose visual features
- Semantic similarity tasks
- Transfer learning with limited labeled data
- Fine-grained visual recognition

### ConvNeXtV2 (Modern Convolutional Network)

**Description:**
- Modernized convolutional architecture
- Fully convolutional masked autoencoding (FCMAE) pre-training
- Competitive with transformers while being more efficient
- Strong performance on dense prediction tasks

**Variants:**
- **convnextv2-tiny**: 768-dim features, ~28M parameters
- **convnextv2-base**: 1024-dim features, ~89M parameters

**Best For:**
- Dense visual features
- Spatial structure preservation
- Efficient inference
- Tasks requiring fine spatial details

## Feature Extraction Process

### Pipeline Flow

```
Image → Load & Convert RGB → Preprocess → Model Forward Pass → Extract Features → NumPy Array
```

### Preprocessing

**DINOv2:**
- Resize to 224×224 (or model-specific size)
- Normalize using ImageNet statistics
- Convert to tensor format

**ConvNeXtV2:**
- Multi-scale cropping and resizing
- Normalization using model-specific statistics
- Data augmentation for inference (center crop)

### Feature Extraction

**DINOv2:**
- Uses [CLS] token from final layer
- Global image representation
- 384-dim (small) or 768-dim (base)

**ConvNeXtV2:**
- Global average pooling of final feature maps
- Dense convolutional features aggregated
- 768-dim (tiny) or 1024-dim (base)

## Batch Processing

The pipeline implements efficient batch processing:

1. **Batching**: Groups images into batches of specified size
2. **GPU Processing**: Processes entire batches on GPU when available
3. **Error Handling**: Falls back to individual processing on batch failure
4. **Progress Tracking**: Shows progress bar with tqdm
5. **Memory Management**: Moves features to CPU after extraction

**Optimal Batch Sizes:**
- **GPU (12GB VRAM)**: 32-64 for small models, 16-32 for base models
- **GPU (8GB VRAM)**: 16-32 for small models, 8-16 for base models
- **CPU**: 1-4 (batch processing less beneficial)

## Output Format

### Single Image Features
```python
features.shape  # (feature_dim,)
# Example: (384,) for dinov2-small
```

### Batch Features
```python
batch_features.shape  # (n_images, feature_dim)
# Example: (100, 384) for 100 images with dinov2-small
```

### Saved Features File (.npz)
```python
{
    'features': array([[...], [...], ...]),      # (n_images, feature_dim)
    'image_paths': array(['path1', 'path2']),    # Image paths
    'model_name': 'dinov2-small',                # Model identifier
    'feature_dim': 384,                          # Feature dimension
    # Optional metadata
    'dataset': 'bloco_suvaco',
    'split': 'train'
}
```

## Complete Pipeline Example

Extracting features for classification:

```python
from pathlib import Path
import numpy as np
from src.core.features.feature_extractor import FeatureExtractionPipeline
from src.core.data.dataset_loader import MultilabelDataset

# Load dataset split
dataset = MultilabelDataset('dataset/splits/split_latest.json')
train_paths, y_train = dataset.get_train_data()
test_paths, y_test = dataset.get_test_data()

# Initialize feature extraction pipeline
pipeline = FeatureExtractionPipeline(
    model_name='dinov2-small',
    device='cuda'
)

# Extract and save training features
print("\nExtracting training features...")
X_train = pipeline.extract_and_save(
    image_paths=train_paths,
    output_file='features/train_dinov2_small.npz',
    batch_size=32,
    metadata={'split': 'train', 'n_samples': len(train_paths)}
)

# Extract and save test features
print("\nExtracting test features...")
X_test = pipeline.extract_and_save(
    image_paths=test_paths,
    output_file='features/test_dinov2_small.npz',
    batch_size=32,
    metadata={'split': 'test', 'n_samples': len(test_paths)}
)

print(f"\n✓ Training features: {X_train.shape}")
print(f"✓ Test features: {X_test.shape}")

# Later: Load features for classification
train_data = FeatureExtractionPipeline.load_features('features/train_dinov2_small.npz')
X_train = train_data['features']
y_train = dataset.get_train_data()[1]  # Labels from dataset

# Train classifier
from src.core.classification.multilabel_classifiers import MultilabelClassifierPipeline

classifier = MultilabelClassifierPipeline(
    classifier_type='xgb',
    categories=dataset.get_categories()
)
classifier.optimize_and_train(X_train, y_train, n_trials=50)
```

## Model Comparison

| Model | Feature Dim | Parameters | Speed (GPU) | Memory | Best Use Case |
|-------|-------------|------------|-------------|--------|---------------|
| dinov2-small | 384 | 22M | Fast | Low | General purpose, fast inference |
| dinov2-base | 768 | 86M | Medium | Medium | Higher accuracy, semantic tasks |
| convnextv2-tiny | 768 | 28M | Fast | Low | Spatial features, efficiency |
| convnextv2-base | 1024 | 89M | Medium | Medium | Dense features, high accuracy |

## Performance Considerations

### Extraction Speed

**Single Image (GPU):**
- DINOv2 Small: ~20-30ms
- DINOv2 Base: ~40-60ms
- ConvNeXtV2 Tiny: ~15-25ms
- ConvNeXtV2 Base: ~35-50ms

**Batch Processing (GPU, batch_size=32):**
- DINOv2 Small: ~500-700ms (64-43 images/sec)
- DINOv2 Base: ~1000-1500ms (32-21 images/sec)
- ConvNeXtV2 Tiny: ~400-600ms (80-53 images/sec)
- ConvNeXtV2 Base: ~900-1300ms (36-25 images/sec)

### Memory Usage

**GPU Memory (batch_size=32):**
- DINOv2 Small: ~2-3 GB
- DINOv2 Base: ~4-6 GB
- ConvNeXtV2 Tiny: ~2-3 GB
- ConvNeXtV2 Base: ~4-6 GB

**Feature Storage (1000 images):**
- 384-dim: ~3 MB (float32)
- 768-dim: ~6 MB (float32)
- 1024-dim: ~8 MB (float32)

### Optimization Tips

**For Speed:**
- Use larger batch sizes (up to GPU memory limit)
- Use smaller models (dinov2-small, convnextv2-tiny)
- Enable GPU if available
- Disable progress bar for maximum speed (`show_progress=False`)

**For Accuracy:**
- Use larger models (dinov2-base, convnextv2-base)
- Higher-dimensional features generally improve classification
- DINOv2 base often gives best results for semantic tasks

**For Memory:**
- Reduce batch size
- Use smaller models
- Process and save in chunks for very large datasets

## Error Handling

The pipeline includes robust error handling:

**Batch Failures:**
- Automatically falls back to individual processing
- Reports which images failed
- Continues processing remaining images

**Individual Failures:**
- Returns zero vectors for failed images
- Logs failures for debugging
- Prevents pipeline crashes

**Image Loading Errors:**
- Handles corrupt images
- Handles unsupported formats
- Handles missing files

**Usage Example:**
```python
# The pipeline handles errors gracefully
features = pipeline.extract_from_paths(
    image_paths=image_paths,
    batch_size=32,
    show_progress=True
)

# Check for zero vectors (failed extractions)
failed_indices = np.where(~features.any(axis=1))[0]
if len(failed_indices) > 0:
    print(f"Warning: {len(failed_indices)} images failed extraction")
    failed_paths = [image_paths[i] for i in failed_indices]
    print("Failed images:", failed_paths)
```

## Integration with Project

This module integrates with the classification pipeline:

### Complete Workflow

1. **Dataset Preparation** → `src/core/dataset/`
2. **Train/Test Split** → `src/scripts/preparation/create_train_test_split.py`
3. **Feature Extraction** → This module
4. **Classification** → `src/core/classification/`
5. **Evaluation** → `src/core/evaluation/`

### Example Integration Script

```python
from src.core.data.dataset_loader import MultilabelDataset
from src.core.features.feature_extractor import FeatureExtractionPipeline
from src.core.classification.multilabel_classifiers import MultilabelClassifierPipeline

# 1. Load dataset
dataset = MultilabelDataset('dataset/splits/split_latest.json')
train_paths, y_train = dataset.get_train_data()
test_paths, y_test = dataset.get_test_data()

# 2. Extract features
pipeline = FeatureExtractionPipeline('dinov2-small', device='cuda')
X_train = pipeline.extract_from_paths(train_paths, batch_size=32)
X_test = pipeline.extract_from_paths(test_paths, batch_size=32)

# 3. Train classifier
classifier = MultilabelClassifierPipeline('xgb', categories=dataset.get_categories())
classifier.optimize_and_train(X_train, y_train, n_trials=50)

# 4. Evaluate
metrics = classifier.evaluate(X_test, y_test)
print(f"Test F1 (macro): {metrics['f1_macro']:.3f}")
```

This module provides the critical bridge between raw images and machine learning models, enabling efficient classification without the need for end-to-end training of large vision models.
