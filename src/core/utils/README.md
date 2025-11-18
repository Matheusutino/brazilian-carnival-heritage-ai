# Utils Module

The `src/core/utils` module provides utility functions for common operations across the project, particularly for image processing and encoding tasks.

## Overview

This module contains helper functions that are used by multiple components of the project. Currently focused on image encoding utilities for LLM (Large Language Model) integration.

## Files

### `image_encoding.py`

Image encoding utilities for preparing images to be sent to vision-language models.

**Key Features:**
- Base64 encoding for API transmission
- Automatic image resizing to control payload size
- Iterative quality reduction to meet size constraints
- Token-budget-aware encoding
- Aspect ratio preservation

**Main Function:**

#### `encode_image_for_llm(image_path, max_dim=500, max_b64_length=45000, min_dim=192, initial_quality=90, min_quality=60)`

Encode an image as base64 while keeping the size within token-friendly bounds for LLM APIs.

**Strategy:**
1. Resize image so largest dimension ≤ `max_dim`
2. Encode as JPEG with `initial_quality`
3. If base64 string exceeds `max_b64_length`:
   - Reduce scale by 15% (multiply by 0.85)
   - Or reduce JPEG quality by 5 points
   - Repeat until within limits or minimum thresholds reached

**Args:**
- `image_path` (Path): Path to the image file
- `max_dim` (int, optional): Maximum allowed dimension (width or height). Default: 500
- `max_b64_length` (int, optional): Maximum base64 string length. Default: 45000
- `min_dim` (int): Minimum dimension before stopping resize. Default: 192
- `initial_quality` (int): Starting JPEG quality (0-100). Default: 90
- `min_quality` (int): Minimum JPEG quality allowed. Default: 60

**Returns:**
- Tuple of (base64_string, final_size)
  - `base64_string` (str): Base64-encoded image
  - `final_size` (tuple): Final image dimensions (width, height)

**Usage Example:**
```python
from pathlib import Path
from src.core.utils.image_encoding import encode_image_for_llm

# Encode image for LLM API
image_path = Path('photo.jpg')
b64_string, final_size = encode_image_for_llm(
    image_path=image_path,
    max_dim=500,
    max_b64_length=45000
)

print(f"Original image encoded to {final_size[0]}x{final_size[1]}")
print(f"Base64 length: {len(b64_string)} characters")

# Use with LLM API
import requests
import json

api_payload = {
    "model": "vision-model",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_string}"}}
            ]
        }
    ]
}

response = requests.post("https://api.example.com/chat", json=api_payload)
```

### Advanced Usage

#### Custom Quality/Size Constraints

```python
from src.core.utils.image_encoding import encode_image_for_llm

# High quality, larger images (for detailed analysis)
b64_large, size_large = encode_image_for_llm(
    image_path='detailed_photo.jpg',
    max_dim=1024,           # Larger maximum dimension
    max_b64_length=100000,  # Allow larger payload
    initial_quality=95,     # Higher initial quality
    min_quality=80          # Higher minimum quality
)

# Low quality, smaller images (for quick processing)
b64_small, size_small = encode_image_for_llm(
    image_path='quick_scan.jpg',
    max_dim=256,            # Smaller maximum dimension
    max_b64_length=20000,   # Smaller payload
    initial_quality=75,     # Lower initial quality
    min_quality=50          # Lower minimum quality
)

print(f"Large: {size_large}, base64 length: {len(b64_large)}")
print(f"Small: {size_small}, base64 length: {len(b64_small)}")
```

#### Batch Encoding

```python
from pathlib import Path
from src.core.utils.image_encoding import encode_image_for_llm

# Encode multiple images
image_paths = list(Path('photos/').glob('*.jpg'))

encoded_images = []
for img_path in image_paths:
    try:
        b64, size = encode_image_for_llm(img_path, max_dim=500)
        encoded_images.append({
            'path': str(img_path),
            'name': img_path.name,
            'base64': b64,
            'size': size,
            'base64_length': len(b64)
        })
        print(f"✓ {img_path.name}: {size[0]}x{size[1]} → {len(b64)} chars")
    except Exception as e:
        print(f"✗ {img_path.name}: {e}")

# Calculate statistics
avg_length = sum(img['base64_length'] for img in encoded_images) / len(encoded_images)
max_length = max(img['base64_length'] for img in encoded_images)
print(f"\nAverage base64 length: {avg_length:.0f}")
print(f"Maximum base64 length: {max_length}")
```

## Encoding Process

### Step-by-Step Algorithm

```
1. Load image and convert to RGB
2. Calculate initial scale: scale = min(1.0, max_dim / max(width, height))
3. Set initial quality = initial_quality
4. Loop:
   a. Resize image to new_width = width * scale, new_height = height * scale
   b. Encode as JPEG with current quality
   c. Convert to base64
   d. If length <= max_b64_length: DONE
   e. If at minimum limits (min_dim and min_quality): DONE (best effort)
   f. If size > min_dim: scale *= 0.85 (reduce size by 15%)
   g. Else if quality > min_quality: quality -= 5 (reduce quality)
   h. Repeat from step 4
```

### Encoding Parameters

**Dimension Constraints:**
- `max_dim=500`: Typical for standard API requests (balances quality and size)
- `max_dim=768`: Higher quality for detailed analysis
- `max_dim=256`: Lower quality for quick classification tasks
- `max_dim=1024`: Maximum quality for fine-grained tasks

**Base64 Length Constraints:**
- `max_b64_length=45000`: ~34KB image, typical API limit
- `max_b64_length=20000`: ~15KB image, faster transmission
- `max_b64_length=100000`: ~75KB image, high quality
- `max_b64_length=None`: No constraint, full quality

**Quality Range:**
- `initial_quality=90`: High quality starting point
- `initial_quality=75`: Moderate quality, faster encoding
- `min_quality=60`: Acceptable minimum for most tasks
- `min_quality=50`: Lower bound for difficult cases

## Size Estimation

Approximate base64 lengths for different configurations:

| Max Dim | Quality | Typical Base64 Length | Approximate File Size |
|---------|---------|----------------------|---------------------|
| 256 | 75 | ~10,000 chars | ~7.5 KB |
| 500 | 90 | ~40,000 chars | ~30 KB |
| 768 | 90 | ~80,000 chars | ~60 KB |
| 1024 | 95 | ~150,000 chars | ~112 KB |

**Formula:** File size ≈ base64_length × 0.75 bytes

## Integration with Project

This utility is used throughout the project for LLM-based image analysis:

### Caption Generation

```python
from pathlib import Path
from src.core.caption.llm_captioner import LLMCaptioner
from src.core.utils.image_encoding import encode_image_for_llm

# The LLMCaptioner uses encode_image_for_llm internally
captioner = LLMCaptioner(
    config_path='config/prompts.yaml',
    provider='nvidia',
    model_name='meta/llama-3.2-90b-vision-instruct',
    max_image_dim=768  # Passed to encode_image_for_llm
)

result = captioner.generate_caption('photo.jpg')
# Image is automatically encoded with max_dim=768
```

### Classification

```python
from pathlib import Path
from src.core.classification.llm_classifier import ImageClassifier

# The ImageClassifier uses encode_image_for_llm internally
classifier = ImageClassifier(
    config_path='config/prompts.yaml',
    categories=['carnival', 'street', 'costume'],
    provider='nvidia',
    model_name='meta/llama-3.2-90b-vision-instruct',
    max_image_dim=500  # Passed to encode_image_for_llm
)

result = classifier.classify('photo.jpg')
# Image is automatically encoded with max_dim=500
```

### Manual API Integration

```python
import json
import requests
from pathlib import Path
from src.core.utils.image_encoding import encode_image_for_llm

# Prepare image
image_path = Path('carnival_photo.jpg')
b64_image, final_size = encode_image_for_llm(
    image_path=image_path,
    max_dim=768,
    max_b64_length=80000
)

# Build API request
api_url = "https://api.nvidia.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": "meta/llama-3.2-90b-vision-instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this Brazilian carnival photograph in detail."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_image}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 500,
    "temperature": 0.7
}

# Send request
response = requests.post(api_url, headers=headers, json=payload)
result = response.json()

print(f"Image size: {final_size[0]}x{final_size[1]}")
print(f"Base64 length: {len(b64_image)}")
print(f"Response: {result['choices'][0]['message']['content']}")
```

## Performance Considerations

### Encoding Speed

**Single Image:**
- Small images (<500x500): ~50-100ms
- Medium images (500-1500): ~100-300ms
- Large images (1500+): ~300-800ms

**Optimization Iterations:**
- Each resize/quality reduction: ~50-100ms
- Typical iterations to meet constraints: 1-5

### Memory Usage

**Peak Memory:**
- Original image in memory
- Resized image in memory
- JPEG buffer in memory

**Approximate:** 3-5× the final JPEG size

### Optimization Tips

**For Speed:**
- Set `max_dim` to target size from start (avoid iterations)
- Use lower `initial_quality` (75 instead of 90)
- Increase `max_b64_length` tolerance

**For Quality:**
- Use higher `max_dim` (768-1024)
- Use higher `initial_quality` (90-95)
- Set higher `min_quality` (70-80)

**For Size:**
- Use lower `max_dim` (256-384)
- Use lower `initial_quality` (70-80)
- Set strict `max_b64_length` (20000-30000)

## Error Handling

### Common Issues

**Invalid Image File:**
```python
try:
    b64, size = encode_image_for_llm('corrupted.jpg')
except Exception as e:
    print(f"Error encoding image: {e}")
```

**Zero-Size Image:**
```python
ValueError: Image has invalid dimensions: /path/to/image.jpg
```
**Solution:** Image file is corrupted or has 0x0 dimensions

**Cannot Meet Size Constraint:**
- Function returns best-effort result (at min_dim and min_quality)
- Base64 length may still exceed `max_b64_length`
- Check returned base64 length and adjust constraints if needed

```python
b64, size = encode_image_for_llm('large_photo.jpg', max_b64_length=10000)
if len(b64) > 10000:
    print(f"Warning: Could not meet size constraint. Got {len(b64)} chars")
    # Either increase max_b64_length or reduce max_dim
```

## Future Extensions

Potential additions to the utils module:

- **Image validation**: Check image integrity before encoding
- **Batch encoding**: Parallel processing for multiple images
- **Format conversion**: PNG → JPEG, HEIC → JPEG, etc.
- **Metadata extraction**: EXIF data, dimensions, file info
- **Image preprocessing**: Normalization, cropping, padding
- **Caching**: Cache encoded images to avoid re-encoding

This utility module provides essential functionality for efficient LLM-vision model integration, optimizing the balance between image quality, API payload size, and processing speed.
