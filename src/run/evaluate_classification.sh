#!/bin/bash
# Evaluate classification outputs (supervised + LLM)

set -e

echo "========================================================================"
echo "EVALUATING CLASSIFICATION RESULTS"
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

SPLIT_FILE="dataset/splits/split_latest.json"
SUP_DIR="results/classification/supervised_classifiers"
LLM_DIR="results/classification/llm"

# -------------------------------------------------------------------------
# Supervised classifiers
# -------------------------------------------------------------------------
if [ -d "$SUP_DIR" ]; then
    echo "------------------------------------------------------------------------"
    echo "Supervised classifiers"
    echo "------------------------------------------------------------------------"

    shopt -s nullglob
    for feature_dir in "$SUP_DIR"/*/; do
        feature_model="$(basename "$feature_dir")"
        echo ""
        echo "Feature extractor: $feature_model"

        for clf_dir in "${feature_dir}"*/; do
            clf_name="$(basename "$clf_dir")"
            predictions_file="${clf_dir}test_predictions.npz"
            metrics_file="${clf_dir}metrics_latest.json"

            if [ -f "$predictions_file" ]; then
                python -m src.scripts.evaluation.evaluate_classifications \
                    --predictions "$predictions_file" \
                    --output "$metrics_file" \
                    --quiet
                rel_path="${metrics_file#$PROJECT_ROOT/}"
                echo "  ✓ ${clf_name}: metrics saved to ${rel_path}"
            else
                echo "  ⚠ ${clf_name}: predictions file not found at $predictions_file"
            fi
        done
    done
    shopt -u nullglob
else
    echo "⚠ No supervised results found at $SUP_DIR"
fi

# -------------------------------------------------------------------------
# LLM classifiers
# -------------------------------------------------------------------------
if [ -d "$LLM_DIR" ]; then
    echo ""
    echo "------------------------------------------------------------------------"
    echo "LLM classifiers"
    echo "------------------------------------------------------------------------"

    shopt -s nullglob
    for model_dir in "$LLM_DIR"/*/; do
        model_name="$(basename "$model_dir")"
        json_files=("${model_dir}"classifications_*.json)
        if [ ${#json_files[@]} -eq 0 ]; then
            echo "  ⚠ ${model_name}: no classification JSON files found"
            continue
        fi

        latest_json="${json_files[0]}"
        for candidate in "${json_files[@]}"; do
            if [ "$candidate" -nt "$latest_json" ]; then
                latest_json="$candidate"
            fi
        done

        metrics_file="${model_dir}metrics_latest.json"
        npz_file="${model_dir}test_predictions_llm.npz"

        python -m src.scripts.evaluation.evaluate_llm_classifications \
            --split "$SPLIT_FILE" \
            --json "$latest_json" \
            --output "$metrics_file" \
            --npz-out "$npz_file"

        rel_metrics="${metrics_file#$PROJECT_ROOT/}"
        echo "  ✓ ${model_name}: metrics saved to ${rel_metrics}"
    done
    shopt -u nullglob
else
    echo ""
    echo "⚠ No LLM results found at $LLM_DIR"
fi
shopt -u nullglob >/dev/null 2>&1 || true

echo ""
echo "========================================================================"
echo "✓ CLASSIFICATION EVALUATION COMPLETE"
echo "========================================================================"
echo ""
