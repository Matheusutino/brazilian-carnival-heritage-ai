# Caption Module

The `src/core/caption` module provides image captioning capabilities using two complementary approaches: classical vision models and multimodal Large Language Models (LLMs).

## Overview

This module offers two distinct captioning strategies:

1. **Classical Vision Models** (`caption_generator.py`) - Fast, efficient captioning using pre-trained vision-language models like BLIP, GIT, and ViT-GPT2
2. **Multimodal LLMs** (`llm_captioner.py`) - Rich, structured captions with detailed cultural and contextual analysis

## Files

### `caption_generator.py`

Image caption generator using classical vision-language models with optional translation to Portuguese.

**Key Features:**
- Supports multiple vision model backends:
  - BLIP (Salesforce/blip-image-captioning-base)
  - GIT (microsoft/git-base)
  - ViT-GPT2 (nlpconnect/vit-gpt2-image-captioning)
- Automatic English-to-Portuguese translation using mBART
- Efficient batch processing with native PyTorch batching
- GPU acceleration support
- Thread-based parallel processing (legacy mode)

**Class:** `CaptionGenerator`

**Usage Example:**
```python
from pathlib import Path
from src.core.caption.caption_generator import CaptionGenerator

# Initialize generator with BLIP backend
generator = CaptionGenerator(
    backend='blip',
    device='cuda',
    translate=True,
    max_length=200
)

# Generate caption for a single image
image_path = Path('image.jpg')
result = generator.generate_caption(image_path)
print(result['caption_en'])  # English caption
print(result['caption_pt'])  # Portuguese translation

# Generate captions for multiple images (efficient batch processing)
image_paths = [Path(f'image_{i}.jpg') for i in range(100)]
results = generator.generate_batch(image_paths, batch_size=8)
```

### `llm_captioner.py`

Image captioner using multimodal LLMs with structured output and retry mechanisms.

**Key Features:**
- Generates rich, structured captions including:
  - Short and long descriptions
  - Keywords for indexing
  - Visual elements analysis (people, setting, objects, colors)
  - Temporal context estimation
  - Cultural analysis (specific to Brazilian carnival culture)
  - Technical quality assessment
- JSON schema validation for output consistency
- Automatic retry mechanism with temperature randomization
- Configurable prompts via YAML configuration
- Base64 image encoding with automatic resizing
- Parallel batch processing with thread pool

**Class:** `LLMCaptioner`

**Usage Example:**
```python
from pathlib import Path
from src.core.caption.llm_captioner import LLMCaptioner

# Initialize captioner with configuration
captioner = LLMCaptioner(
    config_path='config/prompts.yaml',
    provider='nvidia',
    model_name='meta/llama-3.2-90b-vision-instruct',
    temperature=0.7,
    max_tokens=2000,
    max_retries=3,
    prompt_key='caption'
)

# Generate caption for a single image
image_path = Path('carnival_photo.jpg')
result = captioner.generate_caption(image_path)

if result['status'] == 'success':
    caption = result['caption']
    print(caption['descricao_curta'])  # Short description
    print(caption['palavras_chave'])   # Keywords
    print(caption['elementos_visuais']) # Visual elements
    print(caption['contexto_cultural']) # Cultural context

# Generate captions for multiple images in parallel
image_paths = [Path(f'photo_{i}.jpg') for i in range(50)]
results = captioner.generate_batch(image_paths, max_workers=4)
```

### `__init__.py`

Package initialization file.

## Architecture

### Classical Vision Models Pipeline

```
Image → PIL Image → Vision Model → English Caption → (optional) mBART Translation → Portuguese Caption
```

### LLM Captioning Pipeline

```
Image → Base64 Encoding → Prompt Construction → LLM API → JSON Response → Schema Validation → Structured Caption
```

## Configuration

### CaptionGenerator Configuration

```python
CaptionGenerator(
    backend='blip-pt',      # Vision model: 'blip', 'git', or 'vit-gpt2'
    device='cuda',          # Compute device: 'cuda' or 'cpu'
    translate=True,         # Enable Portuguese translation
    translator_model='facebook/mbart-large-50-many-to-many-mmt',
    max_length=200          # Maximum generation length
)
```

### LLMCaptioner Configuration

```python
LLMCaptioner(
    config_path='config/prompts.yaml',  # YAML configuration file
    provider='nvidia',                   # LLM provider
    model_name='meta/llama-3.2-90b-vision-instruct',
    temperature=0.7,                     # Generation temperature
    max_tokens=2000,                     # Maximum output tokens
    max_retries=3,                       # Retry attempts on failure
    retry_temp_min=0.0,                  # Min temperature for retries
    retry_temp_max=1.0,                  # Max temperature for retries
    prompt_key='caption',                # Key in YAML config
    max_image_dim=None                   # Optional image resizing
)
```

### YAML Configuration Structure

The LLMCaptioner requires a YAML configuration file with the following structure:

```yaml
caption:
  system_prompt: |
    System instructions for the LLM...

  user_prompt: |
    User instructions with {image_tag} placeholder...

  output_schema:
    type: object
    required: [descricao_curta, descricao_longa, palavras_chave]
    properties:
      descricao_curta:
        type: string
      descricao_longa:
        type: string
      palavras_chave:
        type: array
        items:
          type: string
      # ... more schema definitions
```

## Output Formats

### CaptionGenerator Output

```python
{
    'image_path': '/path/to/image.jpg',
    'image_name': 'image.jpg',
    'backend': 'blip',
    'caption_en': 'English caption text',
    'caption_pt': 'Portuguese caption text',  # If translate=True
    'caption': 'Main caption',                # PT if translate=True, EN otherwise
    'status': 'success'  # or 'error'
}
```

### LLMCaptioner Output

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
    'caption': {
        'descricao_curta': 'Short description',
        'descricao_longa': 'Detailed description...',
        'palavras_chave': ['keyword1', 'keyword2', ...],
        'elementos_visuais': {
            'pessoas': 'Description of people',
            'cenario': 'Setting description',
            'objetos': 'Objects present',
            'cores_predominantes': ['red', 'blue', 'yellow']
        },
        'contexto_temporal': {
            'epoca_estimada': '1990s',
            'indicadores': 'Temporal indicators'
        },
        'contexto_cultural': {
            'elementos_carnavalescos': 'Carnival elements',
            'tradicoes_identificadas': 'Identified traditions',
            'relevancia_historica': 'Historical relevance'
        },
        'qualidade_tecnica': {
            'nitidez': 'Image sharpness',
            'iluminacao': 'Lighting conditions',
            'estado_conservacao': 'Conservation state'
        }
    },
    'temperature': 0.7,
    'raw_response': 'Raw LLM response text'
}
```

## Performance Considerations

### CaptionGenerator

- **Batch Processing**: Use `generate_batch()` instead of `generate_batch_threaded()` for better performance
  - Native PyTorch batching is 2-5x faster than thread-based processing
  - Optimal batch size depends on GPU memory (typical: 8-16 for CUDA, 1-4 for CPU)
- **GPU Acceleration**: Always use `device='cuda'` when GPU is available
- **Translation**: Disabling translation (`translate=False`) doubles throughput

### LLMCaptioner

- **Parallel Processing**: Use `max_workers` parameter to control thread pool size
  - Typical: 2-8 workers depending on API rate limits and network
- **Retry Strategy**: Adjust `max_retries` and temperature range for reliability vs. speed tradeoff
- **Image Resizing**: Use `max_image_dim` parameter to reduce API payload size and costs

## Error Handling

Both classes include comprehensive error handling:

- Image loading errors are caught and reported in results
- Network and API errors trigger retry mechanisms in LLMCaptioner
- Batch processing continues even if individual images fail
- Status field indicates 'success' or 'error' for each image
