#!/bin/bash
# Upscale all images using all models with scale 2x and 4x

set -e

echo "========================================================================"
echo "UPSCALING ALL IMAGES WITH ALL MODELS (2x and 4x)"
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

# Classical Methods - Scale 2x
echo "========================================================================"
echo "CLASSICAL METHODS - 2x"
echo "========================================================================"

echo ""
echo "[1/13] Nearest Neighbor 2x..."
python -m src.scripts.execution.upscale_images --dataset dataset/dataset_completo.csv --model nearest --scale 2

echo ""
echo "[2/13] Bilinear 2x..."
python -m src.scripts.execution.upscale_images --dataset dataset/dataset_completo.csv --model bilinear --scale 2

echo ""
echo "[3/13] Bicubic 2x..."
python -m src.scripts.execution.upscale_images --dataset dataset/dataset_completo.csv --model bicubic --scale 2

echo ""
echo "[4/13] Lanczos 2x..."
python -m src.scripts.execution.upscale_images --dataset dataset/dataset_completo.csv --model lanczos --scale 2 

# Classical Methods - Scale 4x
echo ""
echo "========================================================================"
echo "CLASSICAL METHODS - 4x"
echo "========================================================================"

echo ""
echo "[5/13] Nearest Neighbor 4x..."
python -m src.scripts.execution.upscale_images --dataset dataset/dataset_completo.csv --model nearest --scale 4

echo ""
echo "[6/13] Bilinear 4x..."
python -m src.scripts.execution.upscale_images --dataset dataset/dataset_completo.csv --model bilinear --scale 4

echo ""
echo "[7/13] Bicubic 4x..."
python -m src.scripts.execution.upscale_images --dataset dataset/dataset_completo.csv --model bicubic --scale 4

echo ""
echo "[8/13] Lanczos 4x..."
python -m src.scripts.execution.upscale_images --dataset dataset/dataset_completo.csv --model lanczos --scale 4 

# Deep Learning Methods - Scale 2x
echo ""
echo "========================================================================"
echo "DEEP LEARNING MODELS - 2x"
echo "========================================================================"

echo ""
echo "[9/13] EDSR 2x..."
python -m src.scripts.execution.upscale_images --dataset dataset/dataset_completo.csv --model edsr --scale 2

echo ""
echo "[10/13] Real-ESRGAN 2x..."
python -m src.scripts.execution.upscale_images --dataset dataset/dataset_completo.csv --model realesrgan --scale 2

# Deep Learning Methods - Scale 4x
echo ""
echo "========================================================================"
echo "DEEP LEARNING MODELS - 4x"
echo "========================================================================"

echo ""
echo "[11/13] EDSR 4x..."
python -m src.scripts.execution.upscale_images --dataset dataset/dataset_completo.csv --model edsr --scale 4

echo ""
echo "[12/13] ESRGAN 4x..."
python -m src.scripts.execution.upscale_images --dataset dataset/dataset_completo.csv --model esrgan --scale 4

echo ""
echo "[13/13] Real-ESRGAN 4x..."
python -m src.scripts.execution.upscale_images --dataset dataset/dataset_completo.csv --model realesrgan --scale 4

# Summary
echo ""
echo "========================================================================"
echo "✓ ALL UPSCALING COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved in: dataset/upscaled/"
echo ""
echo "Models processed:"
echo "  Classical (2x): nearest, bilinear, bicubic, lanczos"
echo "  Classical (4x): nearest, bilinear, bicubic, lanczos"
echo "  Deep Learning (2x): edsr, realesrgan"
echo "  Deep Learning (4x): edsr, esrgan, realesrgan"
echo ""
