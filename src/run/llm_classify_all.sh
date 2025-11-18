#!/bin/bash
# Classify test set images using all LLM models and all prompts

set -e

echo "========================================================================"
echo "TEST SET CLASSIFICATION WITH ALL LLM MODELS AND PROMPTS"
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

# Define all prompts to test
PROMPTS=(
    "classification/zero_shot_basic"
    "classification/zero_shot_with_descriptions"
    "classification/zero_shot_cot"
    "classification/zero_shot_dual_role"
)

# Define models
MODELS=(
    "meta/llama-4-maverick-17b-128e-instruct"
    "meta/llama-4-scout-17b-16e-instruct"
    "mistralai/mistral-small-3.1-24b-instruct-2503"
)

MODEL_NAMES=(
    "Llama 4 Maverick"
    "Llama 4 Scout"
    "Mistral Small"
)

# Counter for progress tracking
TOTAL_RUNS=$((${#MODELS[@]} * ${#PROMPTS[@]}))
CURRENT_RUN=0

# Loop through each model
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_NAME="${MODEL_NAMES[$i]}"

    echo ""
    echo "========================================================================"
    echo "$MODEL_NAME MODEL - ALL PROMPTS"
    echo "========================================================================"

    # Loop through each prompt
    for PROMPT in "${PROMPTS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        PROMPT_NAME=$(echo "$PROMPT" | cut -d'/' -f2)

        echo ""
        echo "[$CURRENT_RUN/$TOTAL_RUNS] $MODEL_NAME - $PROMPT_NAME..."

        # Base command
        CMD="python -m src.scripts.execution.llm_classify_images \
            --split dataset/splits/split_latest.json \
            --config config/prompts.yaml \
            --categories dataset/palavras_chave_processed.csv \
            --output-dir results/classification/llm \
            --provider nvidia \
            --model $MODEL \
            --prompt-key $PROMPT \
            --temperature 0.0 \
            --max-tokens 8000 \
            --max-workers 4 \
            --max-retries 3"

        # Add max-image-dim for Scout model (has image size limitations in NVIDIA API)
        if [[ "$MODEL" == *"scout"* ]]; then
            CMD="$CMD --max-image-dim 512"
        fi

        # Execute command
        eval $CMD
    done
done

# Summary
echo ""
echo "========================================================================"
echo "✓ ALL TEST SET CLASSIFICATION COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved in: results/classification/llm/"
echo ""
echo "Models processed:"
echo "  - Llama 4 Maverick (meta/llama-4-maverick-17b-128e-instruct)"
echo "  - Llama 4 Scout (meta/llama-4-scout-17b-16e-instruct)"
echo "  - Mistral Small (mistralai/mistral-small-3.1-24b-instruct-2503)"
echo ""
echo "Prompts tested:"
echo "  - zero_shot_basic"
echo "  - zero_shot_with_descriptions"
echo "  - zero_shot_cot"
echo "  - zero_shot_dual_role"
echo ""
echo "Total runs: $TOTAL_RUNS (3 models × 4 prompts)"
echo ""
