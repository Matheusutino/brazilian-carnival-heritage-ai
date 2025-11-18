# Pipeline Execution Scripts

This directory contains high-level bash scripts that orchestrate complete processing pipelines for the Bloco Suvaco do Cristo dataset. These scripts automate the execution of multiple models and configurations, making it easy to reproduce experimental results.

## Overview

The scripts are organized into two categories:

1. **Processing Scripts**: Generate outputs (upscaled images, captions, classifications)
2. **Evaluation Scripts**: Calculate metrics for the generated outputs

## Prerequisites

Before running any script, ensure:

```bash
# 1. Virtual environment is created
python3 -m venv venv

# 2. Dependencies are installed
source venv/bin/activate
pip install -r requirements.txt

# 3. Environment variables are set (for LLM models)
export NVIDIA_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"  # if using OpenAI models

# 4. Dataset is prepared
# Ensure dataset/dataset_completo.csv exists and images are in dataset/downloaded/
```

## Processing Scripts

### Image Upscaling

**`upscale_all.sh`** - Upscale images using multiple models and scales

Processes all images with:
- **Classical methods** (2x, 4x): nearest, bilinear, bicubic, lanczos
- **Deep learning models**:
  - 2x: EDSR, Real-ESRGAN
  - 4x: EDSR, ESRGAN, Real-ESRGAN

```bash
./upscale_all.sh
```

**Output**: `dataset/upscaled/<model>_<scale>x/`

**Note**: Images ≥ Full HD (1920x1080) are automatically copied without upscaling to save processing time.

---

### Caption Generation

#### Classical Models

**`caption_classical_all.sh`** - Generate captions using classical vision models

Processes all images with:
- **BLIP** (Salesforce/blip-image-captioning-base)
- **GIT** (microsoft/git-base)
- **ViT-GPT2** (nlpconnect/vit-gpt2-image-captioning)

```bash
./caption_classical_all.sh
```

**Output**: `results/captions/classical/<model>/`

**Features**:
- Automatic English→Portuguese translation
- Batch processing for efficiency
- GPU acceleration when available

#### LLM Models

**`llm_caption_all.sh`** - Generate captions using Vision-Language Models

Processes all images with:
- **Llama 4 Maverick** (meta/llama-4-maverick-17b-128e-instruct)
- **Llama 4 Scout** (meta/llama-4-scout-17b-16e-instruct)
- **Mistral Small** (mistralai/mistral-small-3.1-24b-instruct-2503)

```bash
./llm_caption_all.sh
```

**Output**: `results/captions/llm/<model>/`

**Configuration**:
- Uses prompts from `config/prompts.yaml`
- Parallel processing with 4 workers
- Automatic retry on failures (3 retries)
- Temperature: 0.0 (deterministic)

---

### Image Classification

#### Zero-Shot LLM Classification

**`llm_classify_all.sh`** - Classify images using Vision-Language Models

Classifies test set images with:
- **Llama 4 Maverick**
- **Llama 4 Scout** (with 512px max dimension)
- **Mistral Small**

```bash
./llm_classify_all.sh
```

**Output**: `results/classification/llm/<model>/`

**Requirements**:
- Split file: `dataset/splits/split_latest.json`
- Categories: `dataset/palavras_chave_processed.csv`
- Prompts: `config/prompts.yaml`

#### Supervised Classification

**`supervised_classify_all.sh`** - Train and evaluate supervised classifiers

Pipeline:
1. **Extract visual embeddings** from:
   - DINOv2 (small, base)
   - ConvNeXt V2 (tiny, base)

2. **Train classifiers**:
   - LDA (Linear Discriminant Analysis)
   - Random Forest
   - Decision Tree
   - XGBoost
   - Naive Bayes
   - SVM
   - Passive Aggressive

3. **Optimize hyperparameters** (Optuna, 15 trials, 3-fold CV)

4. **Evaluate on test set**

```bash
./supervised_classify_all.sh
```

**Output**: `results/classification/supervised_classifiers/<feature_model>/<classifier>/`

**Contents**:
- Trained models (`.pkl`)
- Test predictions (`.npz`)
- Evaluation metrics (`.json`)

---

## Evaluation Scripts

### Upscale Evaluation

**`evaluate_upscale.sh`** - Calculate quality metrics for upscaled images

Computes metrics for each upscaling method:
- **No-reference metrics**: NIQE, BRISQUE
- **Full-reference metrics**: PSNR, SSIM (when applicable)

```bash
./evaluate_upscale.sh
```

**Output**: `results/upscale/<method>_metrics.json`

---

### Caption Evaluation

**`evaluate_captions.sh`** - Evaluate caption quality

Evaluates both classical and LLM-generated captions using:
- **BERTScore**: Semantic similarity
- **BLEU**: N-gram precision
- **ROUGE**: Recall-oriented metrics
- **METEOR**: Explicit word matching

```bash
./evaluate_captions.sh
```

**Output**:
- Classical: `results/captions/classical/<model>/caption_metrics_latest.json`
- LLM: `results/captions/llm/<model>/caption_metrics_latest.json`

---

### Classification Evaluation

**`evaluate_classification.sh`** - Calculate classification metrics

Computes comprehensive multilabel classification metrics:
- Per-category precision, recall, F1
- Micro/macro averages
- Sample-wise accuracy
- Hamming loss

```bash
./evaluate_classification.sh
```

**Output**: `<prediction_dir>/metrics_latest.json`

---

## Recommended Execution Order

For a complete experimental pipeline:

```bash
# 1. Upscale images
./upscale_all.sh
./evaluate_upscale.sh

# 2. Generate captions
./caption_classical_all.sh
./llm_caption_all.sh
./evaluate_captions.sh

# 3. Classify images
./supervised_classify_all.sh  # Already includes evaluation
./llm_classify_all.sh
./evaluate_classification.sh  # For LLM classification results
```

## Notes

### Resource Requirements

- **GPU recommended** for:
  - Deep learning upscaling (EDSR, ESRGAN, Real-ESRGAN)
  - Classical captioning (BLIP, GIT, ViT-GPT2)
  - Supervised classification (feature extraction)

- **CPU fallback** available for upscaling when GPU memory is insufficient

- **API keys required** for:
  - LLM caption generation (NVIDIA API)
  - LLM classification (NVIDIA API)

### Processing Time Estimates

For ~877 images:

| Task | Time (GPU) | Time (CPU) |
|------|-----------|-----------|
| Classical upscaling | ~5 min | ~10 min |
| DL upscaling (per model) | ~20-40 min | ~2-4 hours |
| Classical captions | ~15-30 min | ~1-2 hours |
| LLM captions | ~30-60 min | N/A (API) |
| Supervised training | ~10-20 min | ~30-60 min |
| LLM classification | ~20-40 min | N/A (API) |

### Error Handling

All scripts:
- Use `set -e` to fail fast on errors
- Activate virtual environment automatically
- Print progress and status messages
- Save outputs in organized directories

### Customization

To modify parameters:
1. Edit the script variables at the top
2. Or call individual Python scripts with custom arguments

Example:
```bash
python -m src.scripts.execution.upscale_images \
    --dataset dataset/dataset_completo.csv \
    --model realesrgan \
    --scale 2 \
    --device cuda
```

## Troubleshooting

### CUDA Out of Memory

For upscaling, the system automatically:
1. Reduces tile size
2. Falls back to CPU if needed

For other tasks:
- Reduce batch size in the script
- Use smaller models
- Process subsets of the dataset

### Missing Files

Ensure:
- Dataset CSV exists: `dataset/dataset_completo.csv`
- Images exist in: `dataset/downloaded/`
- Split file exists: `dataset/splits/split_latest.json`
- Config file exists: `config/prompts.yaml`
- Categories file exists: `dataset/palavras_chave_processed.csv`

### API Errors

For LLM scripts:
- Verify API keys are set
- Check API quota/limits
- Reduce `--max-workers` for rate limiting
- Increase `--max-retries` for flaky connections

## Output Structure

```
results/
├── upscale/
│   ├── nearest_2x_metrics.json
│   ├── realesrgan_4x_metrics.json
│   └── ...
├── captions/
│   ├── classical/
│   │   ├── blip/
│   │   │   ├── captions_YYYYMMDD_HHMMSS.json
│   │   │   └── caption_metrics_latest.json
│   │   └── ...
│   └── llm/
│       ├── llama-4-maverick/
│       │   ├── captions_YYYYMMDD_HHMMSS.json
│       │   └── caption_metrics_latest.json
│       └── ...
└── classification/
    ├── supervised_classifiers/
    │   ├── dinov2-small/
    │   │   ├── lda/
    │   │   │   ├── model.pkl
    │   │   │   ├── test_predictions.npz
    │   │   │   └── metrics_latest.json
    │   │   └── ...
    │   └── ...
    └── llm/
        ├── llama-4-maverick/
        │   ├── predictions_YYYYMMDD_HHMMSS.npz
        │   └── metrics_latest.json
        └── ...
```
