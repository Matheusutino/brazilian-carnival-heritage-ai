#!/bin/bash
# Generate captions for all images using all classical models

set -e

echo "========================================================================"
echo "CAPTION GENERATION WITH ALL CLASSICAL MODELS"
echo "========================================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root (two levels up from src/run/)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Activate virtual environment
source venv/bin/activate

# BLIP Model
echo "========================================================================"
echo "BLIP MODEL"
echo "========================================================================"

echo ""
echo "[1/3] BLIP - Caption Generation..."
python -m src.scripts.execution.caption_images_classical \
    --dataset dataset/dataset_completo.csv \
    --output-dir results/captions/classical \
    --backend blip \
    --batch-size 8

# GIT Model
echo ""
echo "========================================================================"
echo "GIT MODEL"
echo "========================================================================"

echo ""
echo "[2/3] GIT - Caption Generation..."
python -m src.scripts.execution.caption_images_classical \
    --dataset dataset/dataset_completo.csv \
    --output-dir results/captions/classical \
    --backend git \
    --batch-size 8

# ViT-GPT2 Model
echo ""
echo "========================================================================"
echo "VIT-GPT2 MODEL"
echo "========================================================================"

echo ""
echo "[3/3] ViT-GPT2 - Caption Generation..."
python -m src.scripts.execution.caption_images_classical \
    --dataset dataset/dataset_completo.csv \
    --output-dir results/captions/classical \
    --backend vit-gpt2 \
    --batch-size 8

# Summary
echo ""
echo "========================================================================"
echo "✓ ALL CAPTION GENERATION COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved in: results/captions/classical/"
echo ""
echo "Models processed:"
echo "  - BLIP (Salesforce/blip-image-captioning-base)"
echo "  - GIT (microsoft/git-base)"
echo "  - ViT-GPT2 (nlpconnect/vit-gpt2-image-captioning)"
echo ""
echo "All captions include English→Portuguese translation by default"
echo ""
