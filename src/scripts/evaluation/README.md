# 📊 Evaluation Scripts

This directory contains scripts for evaluating model predictions and generated outputs across three main tasks: **caption generation**, **multilabel classification**, and **image upscaling**. All scripts compute comprehensive metrics and save results in JSON format for analysis.

## 📄 Scripts Overview

| Script | Task | Input Format | Metrics |
|--------|------|--------------|---------|
| `evaluate_caption_outputs.py` | Caption evaluation | JSON (classical/LLM) | BLEU, METEOR, ROUGE-L, BERTScore |
| `evaluate_caption_similarity.py` | Caption evaluation | JSON (simple format) | BLEU, METEOR, ROUGE-L, BERTScore |
| `evaluate_classifications.py` | Classification | NPZ (classical models) | Precision, Recall, F1, Accuracy, Hamming Loss, etc. |
| `evaluate_llm_classifications.py` | Classification | JSON (LLM outputs) | Precision, Recall, F1, Accuracy, Hamming Loss, etc. |
| `evaluate_upscale_quality.py` | Image upscaling | Image files | Sharpness, Edge Strength, Contrast, Artifacts, PSNR |

---

## 🎯 Caption Evaluation Scripts

### 1. `evaluate_caption_outputs.py`

**Purpose**: Evaluate caption generation outputs from both classical models (BLIP, GIT, ViT-GPT2) and LLM pipelines against human-written descriptions.

**Key Features**:
- ✅ **Dual Pipeline Support**: Handles both classical and LLM caption formats
- ✅ **Flexible Field Selection**: Choose which LLM caption field to evaluate (`descricao_longa`, `descricao_curta`, etc.)
- ✅ **Batch Evaluation**: Process multiple output files at once
- ✅ **Portuguese BERT**: Uses `neuralmind/bert-base-portuguese-cased` for BERTScore

**Usage**:
```bash
# Evaluate classical captions only
python -m src.scripts.evaluation.evaluate_caption_outputs \
    --dataset dataset/dataset_completo.csv \
    --classical results/captions/classical/blip/captions_latest.json \
    --output results/captions/blip_metrics.json

# Evaluate LLM captions only
python -m src.scripts.evaluation.evaluate_caption_outputs \
    --dataset dataset/dataset_completo.csv \
    --llm results/captions/llm/llama-4-maverick/caption/captions_20250101.json \
    --llm-preferred-field descricao_longa \
    --output results/captions/llama_metrics.json

# Evaluate both at once
python -m src.scripts.evaluation.evaluate_caption_outputs \
    --dataset dataset/dataset_completo.csv \
    --classical results/captions/classical/blip/captions_latest.json \
                results/captions/classical/git/captions_latest.json \
    --llm results/captions/llm/llama-4-maverick/caption/captions_20250101.json \
         results/captions/llm/llama-4-scout/caption/captions_20250102.json \
    --output results/captions/all_metrics.json
```

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `--dataset` | `str` | Yes | - | Path to dataset CSV with human descriptions |
| `--classical` | `str[]` | No | `[]` | Paths to classical caption JSON outputs |
| `--llm` | `str[]` | No | `[]` | Paths to LLM caption JSON outputs |
| `--llm-preferred-field` | `str` | No | `descricao_longa` | Preferred field in LLM structured captions |
| `--bert-model` | `str` | No | `neuralmind/bert-base-portuguese-cased` | BERT model for BERTScore |
| `--output` | `str` | No | - | JSON file to save aggregated metrics |

**Input Format - Classical**:
```json
[
  {
    "image_path": "dataset/downloaded/img_001.jpg",
    "image_name": "img_001.jpg",
    "caption": "A carnival float with dancers in colorful costumes",
    "status": "success"
  }
]
```

**Input Format - LLM**:
```json
[
  {
    "image_path": "dataset/downloaded/img_001.jpg",
    "image_name": "img_001.jpg",
    "caption": {
      "descricao_longa": "A elaborately decorated carnival float...",
      "descricao_curta": "Carnival float with dancers",
      "observacoes_adicionais": "Evening setting with crowd"
    },
    "status": "success"
  }
]
```

**Metrics Computed**:
- **BLEU-1, BLEU-2, BLEU-3, BLEU-4**: N-gram precision
- **METEOR**: Alignment-based metric with synonyms
- **ROUGE-L**: Longest common subsequence
- **BERTScore**: Contextual embeddings similarity (Precision, Recall, F1)

---

### 2. `evaluate_caption_similarity.py`

**Purpose**: Core caption evaluation script with simpler input format. Used as a utility by `evaluate_caption_outputs.py`.

**Usage**:
```bash
python -m src.scripts.evaluation.evaluate_caption_similarity \
    --csv dataset/dataset_completo.csv \
    --caption-generator results/captions/generated_captions.json \
    --output caption_metrics.json
```

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `--csv` | `str` | Yes | - | Path to dataset CSV |
| `--caption-generator` | `str` | No | - | Path to caption generator JSON |
| `--llm-classifier` | `str` | No | - | Path to LLM classifier JSON |
| `--output` | `str` | No | `caption_evaluation_results.json` | Output JSON file |
| `--bert-model` | `str` | No | `neuralmind/bert-base-portuguese-cased` | BERT model |

**Input Format** (simple):
```json
{
  "img_001.jpg": "A carnival scene with dancers",
  "img_002.jpg": "Traditional costumes in parade"
}
```

or

```json
[
  {"filename": "img_001.jpg", "caption": "A carnival scene with dancers"},
  {"filename": "img_002.jpg", "caption": "Traditional costumes in parade"}
]
```

**Output Format**:
```json
{
  "caption_type": "blip",
  "num_pairs": 300,
  "metrics": {
    "bleu_1": {"mean": 0.45, "std": 0.12},
    "bleu_2": {"mean": 0.32, "std": 0.10},
    "bleu_3": {"mean": 0.24, "std": 0.09},
    "bleu_4": {"mean": 0.18, "std": 0.08},
    "meteor": {"mean": 0.38, "std": 0.11},
    "rouge_l": {"mean": 0.42, "std": 0.10},
    "bertscore_precision": {"mean": 0.78, "std": 0.05},
    "bertscore_recall": {"mean": 0.76, "std": 0.06},
    "bertscore_f1": {"mean": 0.77, "std": 0.05}
  },
  "per_image": [...]
}
```

---

## 🏷️ Classification Evaluation Scripts

### 3. `evaluate_classifications.py`

**Purpose**: Evaluate multilabel classification predictions from classical models (CLIP, SigLIP) stored in NPZ format.

**Key Features**:
- ✅ **Comprehensive Metrics**: Micro, macro, and weighted averages
- ✅ **Per-Image Metrics**: Exact match, Hamming loss, Jaccard score
- ✅ **Per-Category Breakdown**: Detailed metrics for each of the 23 categories
- ✅ **NPZ Format**: Standard format for classical model predictions

**Usage**:
```bash
python -m src.scripts.evaluation.evaluate_classifications \
    --predictions results/classification/classical/clip/test_predictions.npz \
    --output results/classification/classical/clip/metrics.json
```

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `--predictions` | `str` | Yes | - | Path to predictions NPZ file |
| `--output` | `str` | No | - | Output JSON file for metrics |
| `--quiet` | `flag` | No | `False` | Suppress detailed output |

**Input Format (NPZ)**:
```python
{
    'y_true': np.ndarray,      # Shape: (n_samples, 23)
    'y_pred': np.ndarray,      # Shape: (n_samples, 23)
    'categories': np.ndarray,  # List of 23 category names
    'test_paths': np.ndarray   # List of image paths
}
```

**Metrics Computed**:

**Per-Image Metrics**:
- **Exact Match Ratio**: Proportion of images with perfect predictions
- **Hamming Loss**: Average proportion of wrong labels per image
- **Jaccard Score**: Average IoU between true and predicted labels

**Aggregated Metrics** (for each category + overall):
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: (TP + TN) / Total
- **Support**: Number of true instances

**Averaging Methods**:
- **Micro**: Aggregate contributions of all classes (overall performance)
- **Macro**: Unweighted mean of per-class metrics (treat all classes equally)
- **Weighted**: Weighted mean by support (accounts for class imbalance)

**Output Example**:
```
================================================================================
EVALUATION METRICS
================================================================================

Per-Image Metrics:
  Exact Match Ratio:  0.1234
  Hamming Loss:       0.0987
  Jaccard Score:      0.5678

Micro-averaged (overall performance):
  Precision:          0.7845
  Recall:             0.7623
  F1 Score:           0.7732
  Accuracy:           0.9456

Macro-averaged (per-class average):
  Precision:          0.6543
  Recall:             0.6234
  F1 Score:           0.6385
  Accuracy:           0.9234

Top 5 categories by F1:
  fantasia                  F1: 0.8234 (P: 0.821, R: 0.826, Sup: 145)
  pessoas                   F1: 0.8123 (P: 0.798, R: 0.827, Sup: 178)
  ...
```

---

### 4. `evaluate_llm_classifications.py`

**Purpose**: Evaluate multilabel classification outputs from LLM pipelines (JSON format) by converting to NPZ and computing metrics.

**Key Features**:
- ✅ **JSON to NPZ Conversion**: Converts LLM JSON outputs to standard NPZ format
- ✅ **Path Normalization**: Handles different path representations
- ✅ **Missing Prediction Handling**: Treats failed/missing predictions as zeros
- ✅ **Same Metrics as Classical**: Uses identical evaluation logic for fair comparison

**Usage**:
```bash
python -m src.scripts.evaluation.evaluate_llm_classifications \
    --split dataset/splits/split_latest.json \
    --json results/classification/llm/llama-4-maverick/zero_shot_basic/classifications_20250101.json \
    --output results/classification/llm/llama-4-maverick/zero_shot_basic/metrics_latest.json
```

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `--split` | `str` | Yes | - | Train/test split JSON file |
| `--json` | `str` | Yes | - | LLM classification JSON output |
| `--output` | `str` | No | `metrics_latest.json` (same dir) | Metrics JSON output path |
| `--npz-out` | `str` | No | `test_predictions_llm.npz` | Path to save converted NPZ |
| `--quiet` | `flag` | No | `False` | Suppress detailed output |

**Input Format (LLM JSON)**:
```json
[
  {
    "image_path": "dataset/downloaded/img_001.jpg",
    "status": "success",
    "classification": {
      "classificacao_categorias": {
        "adereços": {"valor": 1, "justificativa": "..."},
        "baianas": {"valor": 0, "justificativa": "..."},
        "bateria": {"valor": 1, "justificativa": "..."}
      }
    }
  }
]
```

**Alternative Format** (simplified):
```json
[
  {
    "image_path": "dataset/downloaded/img_001.jpg",
    "status": "success",
    "classificacao_categorias": {
      "adereços": 1,
      "baianas": 0,
      "bateria": 1
    }
  }
]
```

**Binary Value Parsing**:

The script handles multiple value formats:
- `bool`: `True` → 1, `False` → 0
- `int/float`: `≥1` → 1, `<1` → 0
- `str`: `"1"/"true"/"yes"/"sim"/"present"` → 1, others → 0

**Output**:

1. **Metrics JSON**: Same format as `evaluate_classifications.py`
2. **Converted NPZ**: Standard format for further analysis

```json
{
  "predictions_file": "...",
  "n_samples": 300,
  "n_categories": 23,
  "categories": ["adereços", "baianas", ...],
  "missing_predictions": ["img_045.jpg", "img_132.jpg"],
  "per_image_metrics": {...},
  "metrics": {...}
}
```

---

## 🖼️ Image Upscaling Evaluation

### 5. `evaluate_upscale_quality.py`

**Purpose**: Evaluate upscaled image quality using no-reference and self-consistency metrics.

**Key Features**:
- ✅ **No-Reference Metrics**: Sharpness, edge strength, contrast
- ✅ **Artifact Detection**: Over-sharpening, ringing
- ✅ **Self-Consistency**: PSNR with downsampled version (measures preservation)
- ✅ **Batch Processing**: Parallel evaluation with configurable workers
- ✅ **Comprehensive Statistics**: Mean, std, min, max for each metric

**Usage**:
```bash
# Evaluate upscaled images
python -m src.scripts.evaluation.evaluate_upscale_quality \
    --upscaled-dir results/upscale/bicubic_4x \
    --original-dir dataset/downloaded \
    --output-dir results/upscale \
    --n-workers 8

# The script will save: results/upscale/bicubic_4x_metrics.json
```

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `--upscaled-dir` | `str` | Yes | - | Directory with upscaled images |
| `--original-dir` | `str` | No | `dataset/downloaded` | Directory with original images |
| `--output-dir` | `str` | No | `results/upscale` | Output directory for metrics |
| `--n-workers` | `int` | No | `4` | Number of parallel workers |
| `--quiet` | `flag` | No | `False` | Suppress detailed output |

**Metrics Computed**:

| Metric | Description | Interpretation | Range |
|--------|-------------|----------------|-------|
| **Sharpness** | Variance of Laplacian | Higher = sharper image | 0-∞ (higher better) |
| **Edge Strength** | Mean magnitude of Sobel gradients | Higher = stronger edges | 0-255 (higher better) |
| **Contrast** | Standard deviation of pixel values | Higher = more contrast | 0-128 (higher better) |
| **Over-sharpening** | Excessive high-frequency content | Higher = more artifact | 0-∞ (lower better) |
| **Ringing** | High-frequency oscillations near edges | Higher = more artifact | 0-∞ (lower better) |
| **Self-Consistency** | PSNR with downsampled version | Higher = better preservation | 0-∞ (higher better) |

**Metric Details**:

1. **Sharpness** (Variance of Laplacian):
   - Measures the amount of edges and fine details
   - Formula: `Var(∇²I)` where `∇²` is the Laplacian operator
   - Used in autofocus algorithms

2. **Edge Strength** (Sobel):
   - Measures edge magnitude using Sobel operators
   - Formula: `Mean(√(Gx² + Gy²))` where `Gx`, `Gy` are gradients
   - Indicates presence of strong edges

3. **Contrast**:
   - Measures the range of luminance values
   - Formula: `Std(I)` where `I` is grayscale intensity
   - Higher contrast = more dynamic range

4. **Over-sharpening**:
   - Detects excessive sharpening artifacts (halos, edge enhancement)
   - Formula: Based on high-frequency energy in Laplacian
   - Common issue with aggressive sharpening

5. **Ringing**:
   - Detects oscillations near edges (Gibbs phenomenon)
   - Formula: Based on high-frequency content around edges
   - Common with some upscaling methods (e.g., Lanczos)

6. **Self-Consistency** (PSNR):
   - Measures how well the upscaled image preserves original content
   - Formula: `PSNR(downsampled_upscale, original)`
   - **Note**: This metric measures preservation, NOT perceptual quality

**Output Format**:
```json
{
  "timestamp": "2025-10-24T15:30:00",
  "method": "bicubic_4x",
  "upscaled_dir": "results/upscale/bicubic_4x",
  "original_dir": "dataset/downloaded",
  "n_images": 300,
  "n_successful": 295,
  "aggregated_metrics": {
    "sharpness": {
      "mean": 145.23,
      "std": 32.45,
      "min": 67.89,
      "max": 234.56
    },
    "edge_strength": {
      "mean": 23.45,
      "std": 5.67,
      "min": 12.34,
      "max": 45.67
    },
    ...
  },
  "results": [
    {
      "image_path": "...",
      "status": "success",
      "sharpness": 145.23,
      "edge_strength": 23.45,
      ...
    }
  ]
}
```

**Console Output Example**:
```
================================================================================
UPSCALE QUALITY METRICS SUMMARY
================================================================================

Images evaluated: 295/300

Metric                    Mean         Std          Min          Max
--------------------------------------------------------------------------------
Sharpness                 145.23       32.45        67.89        234.56
Edge Strength             23.45        5.67         12.34        45.67
Contrast                  45.67        8.90         23.45        78.90
Over-sharpening           12.34        3.45         5.67         23.45
Ringing                   8.90         2.34         3.45         18.90
Self-Consistency (PSNR)   28.45        3.67         20.12        35.67
```

---

## 🔧 Dependencies

### Caption Evaluation
```bash
pip install pandas nltk rouge-score bert-score transformers torch
```

**NLTK Data** (required for METEOR):
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Classification Evaluation
```bash
pip install numpy pandas scikit-learn
```

### Upscaling Evaluation
```bash
pip install numpy opencv-python pillow scikit-image
```

---

## 📊 Common Workflows

### 1. Evaluate All Caption Models

```bash
# Evaluate all classical models
python -m src.scripts.evaluation.evaluate_caption_outputs \
    --dataset dataset/dataset_completo.csv \
    --classical results/captions/classical/blip/captions_latest.json \
                results/captions/classical/git/captions_latest.json \
                results/captions/classical/vit_gpt2/captions_latest.json \
    --output results/captions/classical_comparison.json

# Evaluate all LLM models
python -m src.scripts.evaluation.evaluate_caption_outputs \
    --dataset dataset/dataset_completo.csv \
    --llm results/captions/llm/llama-4-maverick/caption/captions_*.json \
         results/captions/llm/llama-4-scout/caption/captions_*.json \
         results/captions/llm/mistral-small/caption/captions_*.json \
    --llm-preferred-field descricao_longa \
    --output results/captions/llm_comparison.json
```

### 2. Evaluate All Classification Models

```bash
# Classical models
for model in clip siglip; do
    python -m src.scripts.evaluation.evaluate_classifications \
        --predictions results/classification/classical/${model}/test_predictions.npz \
        --output results/classification/classical/${model}/metrics.json
done

# LLM models (all prompts)
for model in llama-4-maverick llama-4-scout mistral-small; do
    for prompt in zero_shot_basic zero_shot_with_descriptions zero_shot_cot zero_shot_dual_role; do
        python -m src.scripts.evaluation.evaluate_llm_classifications \
            --split dataset/splits/split_latest.json \
            --json results/classification/llm/${model}/${prompt}/classifications_*.json \
            --output results/classification/llm/${model}/${prompt}/metrics_latest.json
    done
done
```

### 3. Evaluate All Upscaling Methods

```bash
# Classical methods
for method in nearest_2x bilinear_2x bicubic_2x lanczos_2x \
              nearest_4x bilinear_4x bicubic_4x lanczos_4x; do
    python -m src.scripts.evaluation.evaluate_upscale_quality \
        --upscaled-dir results/upscale/${method} \
        --original-dir dataset/downloaded \
        --output-dir results/upscale \
        --n-workers 8
done

# Deep learning methods
for method in edsr_2x esrgan_2x realesrgan_2x \
              edsr_4x esrgan_4x realesrgan_4x; do
    python -m src.scripts.evaluation.evaluate_upscale_quality \
        --upscaled-dir results/upscale/${method} \
        --original-dir dataset/downloaded \
        --output-dir results/upscale \
        --n-workers 8
done
```

---

## 🔍 Troubleshooting

### Caption Evaluation

**Error: "No matching pairs found"**

**Problem**: Image filenames don't match between dataset CSV and caption JSON

**Solution**: Check filename format in both files
```bash
# Check dataset CSV
head -n 5 dataset/dataset_completo.csv | cut -d',' -f1

# Check caption JSON
python -c "
import json
with open('results/captions/captions.json') as f:
    data = json.load(f)
    print([item.get('image_name') for item in data[:5]])
"
```

**Error: "BERT model not found"**

**Problem**: BERTScore model not downloaded

**Solution**: Download the model first
```python
from bert_score import BERTScorer
scorer = BERTScorer(model_type="neuralmind/bert-base-portuguese-cased")
```

---

### Classification Evaluation

**Error: "NPZ file not found"**

**Problem**: Predictions file doesn't exist

**Solution**: Run the classifier first
```bash
# For classical models
python -m src.scripts.execution.classify_images_classical \
    --split dataset/splits/split_latest.json \
    --model clip \
    --output-dir results/classification/classical/clip

# For LLM models
python -m src.scripts.execution.llm_classify_images \
    --split dataset/splits/split_latest.json \
    --provider nvidia \
    --model meta/llama-4-maverick-17b-128e-instruct \
    --prompt-key classification/zero_shot_basic \
    --output-dir results/classification/llm
```

**Error: "Shape mismatch"**

**Problem**: NPZ arrays have incompatible shapes

**Solution**: Verify NPZ file structure
```python
import numpy as np
data = np.load('predictions.npz', allow_pickle=True)
print(f"y_true shape: {data['y_true'].shape}")
print(f"y_pred shape: {data['y_pred'].shape}")
print(f"categories: {len(data['categories'])}")
# Should all have same n_labels dimension (23)
```

---

### Upscaling Evaluation

**Error: "No images found"**

**Problem**: Upscaled directory is empty or wrong path

**Solution**: Check directory structure
```bash
ls -lh results/upscale/bicubic_4x/
# Should contain .jpg, .png, etc.
```

**Error: "Original image not found for self-consistency"**

**Problem**: Original image missing or filename mismatch

**Impact**: Self-consistency metric will be `null` for that image

**Solution**:
- Ensure original images are in `--original-dir`
- Upscaled images should have same filenames as originals

**Warning: "Very high over-sharpening detected"**

**Problem**: Aggressive sharpening causing artifacts

**Interpretation**: This is expected for some methods (e.g., ESRGAN)
- Compare with other metrics (sharpness, edge strength)
- High over-sharpening + high sharpness = aggressive enhancement
- High over-sharpening + low sharpness = potential issues

---

## 📈 Metrics Interpretation Guide

### Caption Metrics

**Good Performance**:
- BLEU-4: > 0.30
- METEOR: > 0.40
- ROUGE-L: > 0.45
- BERTScore F1: > 0.75

**Interpretation**:
- **BLEU**: Sensitive to exact n-gram matches (strict)
- **METEOR**: More lenient, considers synonyms
- **ROUGE-L**: Focuses on longest common sequences
- **BERTScore**: Best for semantic similarity (context-aware)

### Classification Metrics

**Good Performance** (multilabel):
- Micro F1: > 0.70
- Macro F1: > 0.60 (harder due to class imbalance)
- Hamming Loss: < 0.10
- Exact Match: > 0.15 (very hard for 23 labels)

**Interpretation**:
- **Micro F1**: Overall performance (treats all predictions equally)
- **Macro F1**: Per-class average (important for rare classes)
- **Hamming Loss**: Average error rate per label
- **Exact Match**: Percentage of perfectly predicted images

### Upscaling Metrics

**Good Performance** (depends on scale factor):

**2x upscaling**:
- Sharpness: > 100
- Edge Strength: > 20
- Contrast: > 40
- Over-sharpening: < 15
- Ringing: < 10
- Self-Consistency (PSNR): > 30 dB

**4x upscaling**:
- Sharpness: > 80
- Edge Strength: > 15
- Contrast: > 35
- Over-sharpening: < 20
- Ringing: < 15
- Self-Consistency (PSNR): > 25 dB

**Trade-offs**:
- **Sharpness vs Over-sharpening**: More sharpness often means more artifacts
- **Edge Strength vs Ringing**: Strong edges may introduce ringing
- **Perceptual Quality vs PSNR**: High PSNR doesn't always mean better perceptual quality

---

## 📚 References

### Caption Metrics
- **BLEU**: Papineni et al. (2002), "BLEU: a Method for Automatic Evaluation of Machine Translation"
- **METEOR**: Banerjee & Lavie (2005), "METEOR: An Automatic Metric for MT Evaluation"
- **ROUGE**: Lin (2004), "ROUGE: A Package for Automatic Evaluation of Summaries"
- **BERTScore**: Zhang et al. (2020), "BERTScore: Evaluating Text Generation with BERT"

### Classification Metrics
- **Multilabel Metrics**: Tsoumakas & Katakis (2007), "Multi-label classification: An overview"
- **Scikit-learn Metrics**: https://scikit-learn.org/stable/modules/model_evaluation.html

### Upscaling Metrics
- **Image Quality Assessment**: Wang & Bovik (2009), "Mean squared error: Love it or leave it?"
- **Sharpness Metrics**: Pech-Pacheco et al. (2000), "Diatom autofocusing in brightfield microscopy"
- **Artifact Detection**: Channappayya et al. (2008), "No-reference image quality assessment"

---

## 🤝 Contributing

To add new evaluation scripts:

1. Follow the naming convention: `evaluate_{task}_{variant}.py`
2. Use consistent argument naming (`--output`, `--quiet`, etc.)
3. Print formatted summaries to console
4. Export results in JSON format with metadata
5. Include comprehensive docstrings and help text
6. Handle edge cases (missing files, empty datasets, errors)
7. Use existing evaluation modules from `src.core.evaluation`
8. Update this README with the new script documentation

---

## 📧 Contact

For questions about evaluation scripts, see the project's main documentation or open an issue in the repository.
