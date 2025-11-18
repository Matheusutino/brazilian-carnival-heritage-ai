#!/bin/bash
# Generate captions for all dataset images using all LLM models

set -e

echo "========================================================================"
echo "CAPTION GENERATION WITH ALL LLM MODELS"
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

# Llama 4 Maverick Model
echo "========================================================================"
echo "LLAMA 4 MAVERICK MODEL"
echo "========================================================================"

echo ""
echo "[1/3] Llama 4 Maverick - Caption Generation..."
python -m src.scripts.execution.llm_generate_captions \
    --dataset dataset/dataset_completo.csv \
    --config config/prompts.yaml \
    --output-dir results/captions/llm \
    --provider nvidia \
    --model meta/llama-4-maverick-17b-128e-instruct \
    --prompt-key caption \
    --temperature 0.0 \
    --max-tokens 8000 \
    --max-workers 4 \
    --max-retries 3

# Llama 4 Scout Model
echo ""
echo "========================================================================"
echo "LLAMA 4 SCOUT MODEL"
echo "========================================================================"

echo ""
echo "[2/3] Llama 4 Scout - Caption Generation..."
python -m src.scripts.execution.llm_generate_captions \
    --dataset dataset/dataset_completo.csv \
    --config config/prompts.yaml \
    --output-dir results/captions/llm \
    --provider nvidia \
    --model meta/llama-4-scout-17b-16e-instruct \
    --prompt-key caption \
    --temperature 0.0 \
    --max-tokens 8000 \
    --max-workers 4 \
    --max-retries 3 \
    --max-image-dim 512 # Necessário, pois o modelo scout tem limitações com imagens maiores ao menos na API da NVIDIA

# Mistral Small Model
echo ""
echo "========================================================================"
echo "MISTRAL SMALL MODEL"
echo "========================================================================"

echo ""
echo "[3/3] Mistral Small - Caption Generation..."
python -m src.scripts.execution.llm_generate_captions \
    --dataset dataset/dataset_completo.csv \
    --config config/prompts.yaml \
    --output-dir results/captions/llm \
    --provider nvidia \
    --model mistralai/mistral-small-3.1-24b-instruct-2503 \
    --prompt-key caption \
    --temperature 0.0 \
    --max-tokens 8000 \
    --max-workers 4 \
    --max-retries 3

# Summary
echo ""
echo "========================================================================"
echo "✓ ALL CAPTION GENERATION COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved in: results/captions/llm/"
echo ""
echo "Models processed:"
echo "  - Llama 4 Maverick (meta/llama-4-maverick-17b-128e-instruct)"
echo "  - Llama 4 Scout (meta/llama-4-scout-17b-16e-instruct)"
echo "  - Mistral Small (mistralai/mistral-small-3.1-24b-instruct-2503)"
echo ""
