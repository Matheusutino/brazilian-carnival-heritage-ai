# Upscaler Module

The `src/core/upscaler` module provides image upscaling capabilities using state-of-the-art deep learning models and classical interpolation methods. It enables high-quality image super-resolution for enhancing low-resolution historical photographs.

## Overview

This module offers a unified interface for multiple upscaling approaches:

1. **Deep Learning Models** - GAN-based and transformer-based super-resolution
   - Real-ESRGAN (practical image restoration)
   - ESRGAN (enhanced super-resolution GAN)
   - SwinIR (Swin Transformer for image restoration)
   - EDSR (enhanced deep residual networks)

2. **Classical Methods** - Traditional interpolation algorithms
   - Nearest neighbor
   - Bilinear
   - Bicubic
   - Lanczos

## Files

### `image_upscaler.py`

Unified image upscaler with support for multiple backends and automatic GPU memory management.

**Key Features:**
- Multiple upscaling models (4 deep learning + 4 classical methods)
- Automatic GPU/CPU device selection
- Tile-based processing for large images
- Dynamic tile size adjustment on GPU out-of-memory
- Automatic CPU fallback for memory-intensive images
- Batch processing with progress tracking
- FP16 (half precision) inference on GPU for faster processing

**Class:** `ImageUpscaler`

### `__init__.py`

Package initialization file.

## Supported Models

### Deep Learning Models

#### Real-ESRGAN

**Description:**
- Practical image restoration algorithm for real-world images
- Extends ESRGAN with improved degradation modeling
- Best for general-purpose upscaling with realistic results
- Handles various degradations (blur, noise, compression)

**Scales:** 2x, 4x

**Models:**
- `RealESRGAN_x2plus` (2x upscaling, 23 RRDB blocks)
- `RealESRGAN_x4plus` (4x upscaling, 23 RRDB blocks)

#### ESRGAN

**Description:**
- Enhanced Super-Resolution GAN
- Original ESRGAN model from the paper
- Perceptual loss for realistic textures
- Residual-in-Residual Dense Blocks (RRDB)

**Scales:** 4x

**Model:** `ESRGAN_SRx4_DF2KOST_official`

#### SwinIR

**Description:**
- Swin Transformer-based image restoration
- Attention mechanism for long-range dependencies
- Better quality than CNN-based methods
- More computationally intensive

**Scales:** 2x, 4x

**Models:**
- `SwinIR-M_x2` (2x upscaling, 6 layers)
- `SwinIR-M_x4` (4x upscaling, 6 layers)

#### EDSR

**Description:**
- Enhanced Deep Residual Networks
- Very deep residual architecture
- State-of-the-art performance on benchmarks
- From HuggingFace super-image library

**Scales:** 2x, 3x, 4x

**Model:** `eugenesiow/edsr-base`

### Classical Methods

Fast interpolation methods that don't require model loading:

- **Nearest**: Nearest neighbor (fastest, blocky results)
- **Bilinear**: Linear interpolation (fast, smooth but blurry)
- **Bicubic**: Cubic interpolation (good quality, fast)
- **Lanczos**: Lanczos resampling (best classical method, slower)

**Scales:** Any positive integer

## Usage

### Basic Usage

```python
from pathlib import Path
from src.core.upscaler.image_upscaler import ImageUpscaler

# Initialize upscaler with Real-ESRGAN
upscaler = ImageUpscaler(
    model='realesrgan',
    scale=4,
    device='cuda',
    tile_size=512,
    tile_pad=10
)

# Upscale single image
result = upscaler.upscale_image(
    image_path='photo.jpg',
    output_path='photo_upscaled.jpg'
)

if result['status'] == 'success':
    print(f"✓ Upscaled: {result['original_size']} → {result['upscaled_size']}")
    print(f"  Saved to: {result['output_path']}")
else:
    print(f"✗ Error: {result['error']}")
```

### Batch Processing

```python
from pathlib import Path
from src.core.upscaler.image_upscaler import ImageUpscaler

# Initialize upscaler
upscaler = ImageUpscaler(
    model='realesrgan',
    scale=4,
    device='cuda'
)

# Get list of images
image_paths = list(Path('dataset/downloaded').glob('*.jpg'))

# Upscale batch
results = upscaler.upscale_batch(
    image_paths=image_paths,
    output_dir='dataset/upscaled_4x',
    verbose=True
)

# Print summary
success = sum(1 for r in results if r['status'] == 'success')
print(f"\nCompleted: {success}/{len(results)} successful")
```

### Using Different Models

```python
from src.core.upscaler.image_upscaler import ImageUpscaler

# Real-ESRGAN 2x
upscaler_2x = ImageUpscaler(model='realesrgan', scale=2)

# ESRGAN 4x
upscaler_esrgan = ImageUpscaler(model='esrgan', scale=4)

# SwinIR 4x (transformer-based)
upscaler_swinir = ImageUpscaler(model='swinir', scale=4)

# EDSR 3x (supports 3x scaling)
upscaler_edsr = ImageUpscaler(model='edsr', scale=3)

# Classical bicubic (fast baseline)
upscaler_bicubic = ImageUpscaler(model='bicubic', scale=4)

# Upscale with each
for upscaler in [upscaler_2x, upscaler_esrgan, upscaler_swinir, upscaler_edsr, upscaler_bicubic]:
    result = upscaler.upscale_image('photo.jpg')
    print(f"{upscaler.model_name}: {result['status']}")
```

### Custom Output Paths

```python
# Auto-generate output path (creates subdirectory)
result = upscaler.upscale_image('photo.jpg')
# Output: photo_upscaled_realesrgan_4x/photo_realesrgan_4x.jpg

# Specify output path explicitly
result = upscaler.upscale_image(
    image_path='photo.jpg',
    output_path='results/photo_enhanced.jpg'
)
```

### CPU-Only Processing

```python
# Force CPU processing (useful for systems without GPU)
upscaler = ImageUpscaler(
    model='realesrgan',
    scale=4,
    device='cpu'
)

result = upscaler.upscale_image('photo.jpg')
```

## Advanced Features

### Tile-Based Processing

For large images that don't fit in GPU memory, the upscaler automatically processes images in tiles:

```python
upscaler = ImageUpscaler(
    model='realesrgan',
    scale=4,
    device='cuda',
    tile_size=512,  # Process in 512x512 tiles
    tile_pad=10     # Padding to avoid edge artifacts
)
```

**Tile Size Guidelines:**
- **512**: Default, works for most images on 8GB+ GPU
- **256**: For 4GB GPU or very large images
- **0**: Disable tiling (process entire image at once)

### Automatic Memory Management

The upscaler includes intelligent memory management:

1. **Dynamic Tile Adjustment**: Automatically reduces tile size on GPU OOM
2. **CPU Fallback**: Falls back to CPU if GPU tiling fails
3. **GPU Restoration**: Moves model back to GPU after CPU processing

```python
# The upscaler will automatically:
# 1. Try full image on GPU
# 2. If OOM: Try with tile_size=512
# 3. If OOM: Try with tile_size=256
# 4. If OOM: Try with tile_size=128
# 5. If still OOM: Process on CPU
# 6. Restore model to GPU for next image

upscaler = ImageUpscaler(model='realesrgan', scale=4, device='cuda')
result = upscaler.upscale_image('very_large_photo.jpg')

if result.get('fallback_cpu'):
    print("⚠ This image was processed on CPU due to GPU memory limitations")
```

## Output Format

### Single Image Result

```python
{
    'image_path': '/path/to/photo.jpg',
    'image_name': 'photo.jpg',
    'model': 'realesrgan',
    'scale': 4,
    'status': 'success',
    'original_size': (800, 600),      # (width, height)
    'upscaled_size': (3200, 2400),    # (width, height)
    'output_path': '/path/to/photo_realesrgan_4x.jpg',
    'fallback_cpu': False             # True if CPU fallback was used
}
```

### Error Result

```python
{
    'image_path': '/path/to/photo.jpg',
    'image_name': 'photo.jpg',
    'model': 'realesrgan',
    'scale': 4,
    'status': 'error',
    'error': 'Failed to read image: /path/to/photo.jpg'
}
```

## Model Comparison

| Model | Scales | Quality | Speed | GPU Memory | Best For |
|-------|--------|---------|-------|------------|----------|
| **Real-ESRGAN** | 2x, 4x | Excellent | Fast | Medium | General purpose, realistic textures |
| **ESRGAN** | 4x | Very Good | Fast | Medium | Sharp details, original ESRGAN |
| **SwinIR** | 2x, 4x | Excellent | Slow | High | Best quality, attention mechanism |
| **EDSR** | 2x, 3x, 4x | Excellent | Medium | Medium | Versatile, supports 3x scaling |
| **Bicubic** | Any | Fair | Very Fast | None | Fast baseline, no GPU needed |
| **Lanczos** | Any | Good | Very Fast | None | Best classical method |

### Quality Ranking (Best to Good)

1. **SwinIR** - Best overall quality, especially for fine details
2. **Real-ESRGAN** - Best for real-world photos with degradations
3. **EDSR** - Excellent quality, good all-rounder
4. **ESRGAN** - Strong sharpness, good textures
5. **Lanczos** - Best classical method
6. **Bicubic** - Good classical baseline
7. **Bilinear** - Basic smoothing
8. **Nearest** - Pixel duplication (blocky)

### Speed Ranking (Fastest to Slowest)

1. **Classical methods** (nearest, bilinear, bicubic, lanczos)
2. **ESRGAN / Real-ESRGAN**
3. **EDSR**
4. **SwinIR**

## Performance Considerations

### Processing Speed

**Single Image (GPU, no tiling):**
- Real-ESRGAN 4x: ~1-3 seconds (1024x768 input)
- ESRGAN 4x: ~1-3 seconds
- SwinIR 4x: ~5-10 seconds
- EDSR 4x: ~2-4 seconds
- Bicubic: ~0.05 seconds

**With Tiling:**
- Speed depends on tile count
- 2x slower for 4 tiles, 4x slower for 16 tiles
- Overhead from tile merging

### Memory Usage

**GPU Memory (approximate):**
- Real-ESRGAN/ESRGAN: 2-4 GB base + image size
- SwinIR: 3-5 GB base + image size
- EDSR: 2-4 GB base + image size

**Image Size Limits (no tiling):**
- 8GB GPU: ~2048x2048 input (4x upscale)
- 12GB GPU: ~3000x3000 input (4x upscale)
- 16GB+ GPU: ~4000x4000 input (4x upscale)

**With Tiling:**
- Can process arbitrarily large images
- Limited only by output size (max ~16000x16000)

### Optimization Tips

**For Speed:**
- Use Real-ESRGAN or ESRGAN (fastest deep learning models)
- Use classical methods for baseline
- Disable tiling for small images (`tile_size=0`)
- Use larger batch sizes (processes sequentially but organized)

**For Quality:**
- Use SwinIR for best quality
- Use Real-ESRGAN for real-world photos
- Use 2x upscaling twice instead of 4x for even better quality

**For Memory:**
- Enable tiling with smaller tile sizes
- Use `tile_size=256` or `tile_size=128` for very large images
- Let automatic CPU fallback handle extreme cases

## Error Handling

### Common Issues

**CUDA Out of Memory:**
```
RuntimeError: CUDA out of memory
```
**Solution:** The upscaler automatically handles this by:
1. Reducing tile size progressively
2. Falling back to CPU if needed

**Model Loading Errors:**
```
ImportError: Real-ESRGAN dependencies not found
```
**Solution:** Install required dependencies:
```bash
pip install realesrgan basicsr
pip install super-image  # For EDSR
```

**Image Loading Errors:**
```
ValueError: Failed to read image
```
**Solution:** Check that:
- Image file exists
- Image format is supported (JPG, PNG, etc.)
- Image is not corrupted

## Complete Example

Full workflow with multiple models and quality assessment:

```python
from pathlib import Path
from src.core.upscaler.image_upscaler import ImageUpscaler
from src.core.evaluation.upscale_metrics import ImageQualityEvaluator

# Input images
image_paths = list(Path('dataset/downloaded').glob('*.jpg'))

# Models to compare
models = [
    ('bicubic', 4),
    ('realesrgan', 4),
    ('swinir', 4)
]

# Upscale with each model
for model_name, scale in models:
    print(f"\n{'='*60}")
    print(f"Upscaling with {model_name} ({scale}x)")
    print(f"{'='*60}")

    # Initialize upscaler
    upscaler = ImageUpscaler(
        model=model_name,
        scale=scale,
        device='cuda'
    )

    # Output directory
    output_dir = f'results/upscaling/{model_name}_{scale}x'

    # Upscale batch
    results = upscaler.upscale_batch(
        image_paths=image_paths,
        output_dir=output_dir,
        verbose=True
    )

    # Report
    success = sum(1 for r in results if r['status'] == 'success')
    print(f"\n✓ {model_name}: {success}/{len(results)} successful")

# Evaluate quality
print(f"\n{'='*60}")
print("Evaluating Quality")
print(f"{'='*60}")

evaluator = ImageQualityEvaluator(original_images_dir='dataset/downloaded')
comparison = evaluator.compare_methods(
    base_dir='results/upscaling',
    verbose=True,
    n_workers=4
)

# Print comparison
print(f"\n{'='*60}")
print("Quality Comparison")
print(f"{'='*60}\n")

for method_name, stats in comparison.items():
    print(f"{method_name}:")
    print(f"  Sharpness: {stats['sharpness_mean']:.2f} ± {stats['sharpness_std']:.2f}")
    print(f"  Edge Strength: {stats['edge_strength_mean']:.2f} ± {stats['edge_strength_std']:.2f}")
    print(f"  Contrast: {stats['contrast_mean']:.2f} ± {stats['contrast_std']:.2f}")
    if stats['self_consistency_mean']:
        print(f"  Self-Consistency (PSNR): {stats['self_consistency_mean']:.2f} dB")
    print()
```

## Integration with Project

This module is used in the Bloco Suvaco de Cristo project for enhancing low-resolution historical photographs:

### Typical Workflow

1. **Dataset Preparation** → `src/core/dataset/`
2. **Image Upscaling** → This module
3. **Quality Evaluation** → `src/core/evaluation/upscale_metrics.py`
4. **Visual Inspection** → Compare original vs upscaled

### Batch Upscaling Script

```python
from pathlib import Path
from src.core.upscaler.image_upscaler import ImageUpscaler
import json

# Configuration
config = {
    'input_dir': 'dataset/downloaded',
    'output_base': 'dataset/upscaled',
    'models': [
        {'name': 'realesrgan', 'scale': 2},
        {'name': 'realesrgan', 'scale': 4},
        {'name': 'bicubic', 'scale': 4}  # Baseline
    ]
}

# Get all images
image_paths = []
for ext in ['.jpg', '.jpeg', '.png']:
    image_paths.extend(Path(config['input_dir']).glob(f'*{ext}'))

print(f"Found {len(image_paths)} images")

# Process with each model
all_results = {}

for model_config in config['models']:
    model_name = model_config['name']
    scale = model_config['scale']

    print(f"\n{'='*60}")
    print(f"Processing: {model_name} {scale}x")
    print(f"{'='*60}")

    # Initialize upscaler
    upscaler = ImageUpscaler(
        model=model_name,
        scale=scale,
        device='cuda'
    )

    # Output directory
    output_dir = Path(config['output_base']) / f'{model_name}_{scale}x'

    # Process batch
    results = upscaler.upscale_batch(
        image_paths=image_paths,
        output_dir=output_dir,
        verbose=True
    )

    # Store results
    all_results[f'{model_name}_{scale}x'] = results

    # Save results
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {results_file}")

print(f"\n{'='*60}")
print("All upscaling complete!")
print(f"{'='*60}")
```

This module provides robust and efficient image upscaling for enhancing historical archive photographs with minimal manual intervention.
