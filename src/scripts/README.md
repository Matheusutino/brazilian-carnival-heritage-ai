# Scripts

This directory contains all command-line scripts for the Bloco Suvaco de Cristo project, organized into two main categories:

## 📂 Directory Structure

```
src/scripts/
├── execution/          # Scripts that perform actual processing operations
│   ├── caption_images_classical.py
│   ├── llm_classify_images.py
│   ├── prepare_dataset.py
│   ├── upscale_images.py
│   └── zero_shot_classify_images.py
└── evaluation/         # Scripts that evaluate and assess results
    ├── evaluate_caption_similarity.py
    └── evaluate_quality.py
```

## 🚀 Execution Scripts

Scripts that **perform processing operations** on images and data:

### `execution/caption_images_classical.py`
Generate image captions using classical vision-language models (BLIP, GIT, ViT-GPT2).

```bash
python src/scripts/execution/caption_images_classical.py \
  --dataset dataset/dataset_completo.csv \
  --backend blip \
  --batch-size 8
```

### `execution/upscale_images.py`
Upscale images using various methods (classical interpolation or deep learning).

```bash
python src/scripts/execution/upscale_images.py \
  --input-dir dataset/downloaded \
  --model bicubic \
  --scale 2 \
  --output-dir results/upscaled
```

### `execution/llm_classify_images.py`
Classify images using Large Language Models with vision capabilities.

```bash
python src/scripts/execution/llm_classify_images.py \
  --dataset dataset/dataset_completo.csv
```

### `execution/zero_shot_classify_images.py`
Zero-shot image classification using CLIP or similar models.

```bash
python src/scripts/execution/zero_shot_classify_images.py \
  --dataset dataset/dataset_completo.csv
```

### `execution/prepare_dataset.py`
Prepare and process the dataset (extract hyperlinks, download files, merge data).

```bash
python src/scripts/execution/prepare_dataset.py
```

## 📊 Evaluation Scripts

Scripts that **evaluate and assess the quality** of processing results:

### `evaluation/evaluate_quality.py`
Evaluate upscaling quality using various metrics (sharpness, edge strength, artifacts, etc.).

```bash
# Compare all upscaling methods
python src/scripts/evaluation/evaluate_quality.py \
  --compare-dir results/upscaled

# Evaluate single method
python src/scripts/evaluation/evaluate_quality.py \
  --input-dir results/upscaled/bicubic_2x
```

### `evaluation/evaluate_caption_similarity.py`
Evaluate caption similarity and quality metrics.

```bash
python src/scripts/evaluation/evaluate_caption_similarity.py \
  --captions results/caption/blip/captions_*.json
```

## 🎯 Quick Reference

### Running All Operations

Use the batch scripts in `src/run/`:

```bash
# Generate captions with all models
bash src/run/caption_classical_all.sh

# Upscale with all models
bash src/run/upscale_all.sh
```

### Common Workflows

**1. Full Pipeline:**
```bash
# Prepare dataset
python src/scripts/execution/prepare_dataset.py

# Generate captions
python src/scripts/execution/caption_images_classical.py --backend blip

# Upscale images
python src/scripts/execution/upscale_images.py --model bicubic --scale 2

# Evaluate quality
python src/scripts/evaluation/evaluate_quality.py --compare-dir results/upscaled
```

**2. Quick Test:**
```bash
# Test captioning on 20 images
python src/scripts/execution/caption_images_classical.py --limit 20

# Evaluate single upscale method
python src/scripts/evaluation/evaluate_quality.py --input-dir results/upscaled/bicubic_2x
```

## 📝 Additional Documentation

- **Batch Processing:** See `BATCH_PROCESSING_README.md` for detailed batch processing guide
- **Upscale Metrics:** See `../docs/UPSCALE_METRICS.md` for evaluation metrics documentation

## 💡 Tips

- Use `--help` on any script to see all available options
- Scripts auto-detect CUDA when available
- Most scripts support parallel processing for better performance
- Output files include timestamps to avoid overwriting
