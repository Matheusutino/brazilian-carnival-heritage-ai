#!/bin/bash
# Train and evaluate supervised multilabel classifiers using visual embeddings

set -e

echo "========================================================================"
echo "SUPERVISED CLASSIFICATION WITH EMBEDDING FEATURES"
echo "========================================================================"
echo ""

# Determine project root (two levels above this script)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Move to project root
cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "✗ Virtual environment not found at venv/. Please create/activate before running."
    exit 1
fi

# Configuration
SPLIT_FILE="dataset/splits/split_latest.json"
OUTPUT_DIR="results/classification/supervised_classifiers"
FEATURE_MODELS=("dinov2-small" "dinov2-base" "convnextv2-tiny" "convnextv2-base")
CLASSIFIERS=("lda" "rf" "tree" "xgb" "nb" "svm" "passive_aggressive")
BATCH_SIZE=32
OPTIMIZE="--optimize"
N_TRIALS=15
CV_FOLDS=3

total_models=${#FEATURE_MODELS[@]}

echo "Using split file ..........: $SPLIT_FILE"
echo "Output directory ..........: $OUTPUT_DIR"
echo "Feature models ............: ${FEATURE_MODELS[*]}"
echo "Supervised classifiers ....: ${CLASSIFIERS[*]}"
echo "Batch size (feature extr.).: $BATCH_SIZE"
echo ""

# Loop through feature models
for idx in "${!FEATURE_MODELS[@]}"; do
    feature_model="${FEATURE_MODELS[$idx]}"
    job_num=$((idx + 1))

    echo "------------------------------------------------------------------------"
    echo "[$job_num/$total_models] Feature extractor: $feature_model"
    echo "------------------------------------------------------------------------"

    python -m src.scripts.execution.train_supervised_classifiers \
        --split "$SPLIT_FILE" \
        --feature-model "$feature_model" \
        --classifiers "${CLASSIFIERS[@]}" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size "$BATCH_SIZE" \
        $OPTIMIZE \
        --n-trials "$N_TRIALS" \
        --cv-folds "$CV_FOLDS"

    echo ""
    echo "Evaluating classifiers for $feature_model..."

    for clf in "${CLASSIFIERS[@]}"; do
        predictions_file="$OUTPUT_DIR/$feature_model/$clf/test_predictions.npz"
        metrics_file="$OUTPUT_DIR/$feature_model/$clf/metrics_latest.json"

        if [ -f "$predictions_file" ]; then
            python -m src.scripts.evaluation.evaluate_classifications \
                --predictions "$predictions_file" \
                --output "$metrics_file" \
                --quiet
            rel_path="${metrics_file#$PROJECT_ROOT/}"
            echo "  ✓ ${clf}: metrics saved to ${rel_path}"
        else
            echo "  ⚠ ${clf}: predictions file not found at $predictions_file"
        fi
    done

    echo ""
done

echo "========================================================================"
echo "✓ SUPERVISED CLASSIFICATION PIPELINE COMPLETE"
echo "========================================================================"
echo ""
echo "Predictions, models and metrics stored under: $OUTPUT_DIR"
echo ""
