#!/bin/bash
# Evaluate caption generation outputs (classical + LLM)

set -e

echo "========================================================================"
echo "EVALUATING CAPTION RESULTS"
echo "========================================================================"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_ROOT"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "✗ Virtual environment not found at venv/. Please create/activate before running."
    exit 1
fi

DATASET_CSV="dataset/dataset_completo.csv"
CLASSICAL_DIR="results/captions/classical"
LLM_DIR="results/captions/llm"

all_evaluated=0

# Evaluate classical backends
if [ -d "$CLASSICAL_DIR" ]; then
    shopt -s nullglob
    for backend_dir in "$CLASSICAL_DIR"/*/; do
        backend_name="$(basename "$backend_dir")"
        files=("${backend_dir}"captions_*.json)
        if [ ${#files[@]} -eq 0 ]; then
            echo "⚠ ${backend_name}: no caption outputs found"
            continue
        fi
        latest="${files[0]}"
        for candidate in "${files[@]}"; do
            if [ "$candidate" -nt "$latest" ]; then
                latest="$candidate"
            fi
        done

        OUTPUT_FILE="${backend_dir}caption_metrics_latest.json"

        rel="${latest#$PROJECT_ROOT/}"
        echo "✓ Classical backend '${backend_name}' → using ${rel}"

        echo ""
        echo "------------------------------------------------------------------------"
        echo "Evaluating ${backend_name}..."
        echo "------------------------------------------------------------------------"
        python -m src.scripts.evaluation.evaluate_caption_outputs \
            --dataset "$DATASET_CSV" \
            --classical "$latest" \
            --output "$OUTPUT_FILE"

        rel_out="${OUTPUT_FILE#$PROJECT_ROOT/}"
        echo "✓ Metrics saved to ${rel_out}"
        echo ""

        all_evaluated=$((all_evaluated + 1))
    done
    shopt -u nullglob
else
    echo "⚠ Classical caption directory not found at $CLASSICAL_DIR"
fi

# Evaluate LLM models
if [ -d "$LLM_DIR" ]; then
    shopt -s nullglob
    for model_dir in "$LLM_DIR"/*/; do
        model_name="$(basename "$model_dir")"
        files=("${model_dir}"captions_*.json)
        if [ ${#files[@]} -eq 0 ]; then
            echo "⚠ ${model_name}: no caption outputs found"
            continue
        fi
        latest="${files[0]}"
        for candidate in "${files[@]}"; do
            if [ "$candidate" -nt "$latest" ]; then
                latest="$candidate"
            fi
        done

        OUTPUT_FILE="${model_dir}caption_metrics_latest.json"

        rel="${latest#$PROJECT_ROOT/}"
        echo "✓ LLM model '${model_name}' → using ${rel}"

        echo ""
        echo "------------------------------------------------------------------------"
        echo "Evaluating ${model_name}..."
        echo "------------------------------------------------------------------------"
        python -m src.scripts.evaluation.evaluate_caption_outputs \
            --dataset "$DATASET_CSV" \
            --llm "$latest" \
            --output "$OUTPUT_FILE"

        rel_out="${OUTPUT_FILE#$PROJECT_ROOT/}"
        echo "✓ Metrics saved to ${rel_out}"
        echo ""

        all_evaluated=$((all_evaluated + 1))
    done
    shopt -u nullglob
else
    echo "⚠ LLM caption directory not found at $LLM_DIR"
fi

if [ $all_evaluated -eq 0 ]; then
    echo "✗ No caption outputs found to evaluate."
    exit 1
fi

echo "✓ Evaluated ${all_evaluated} model(s)"
echo ""
echo "========================================================================"
echo "✓ CAPTION EVALUATION COMPLETE"
echo "========================================================================"
echo ""
