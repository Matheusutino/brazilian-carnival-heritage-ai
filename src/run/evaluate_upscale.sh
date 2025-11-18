#!/bin/bash
# Evaluate quality metrics for all upscaled image directories.

set -e

echo "========================================================================"
echo "EVALUATING UPSCALE RESULTS"
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

UPSCALED_DIR="dataset/upscaled"
ORIGINAL_DIR="dataset/downloaded"
OUTPUT_DIR="results/upscale"

if [ ! -d "$UPSCALED_DIR" ]; then
    echo "✗ Upscaled directory not found at $UPSCALED_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

shopt -s nullglob
upscaled_sets=("$UPSCALED_DIR"/*/)
shopt -u nullglob

if [ ${#upscaled_sets[@]} -eq 0 ]; then
    echo "✗ No upscaled image directories found in $UPSCALED_DIR"
    exit 1
fi

for set_dir in "${upscaled_sets[@]}"; do
    method_name="$(basename "$set_dir")"
    echo "------------------------------------------------------------------------"
    echo "Evaluating: ${method_name}"
    echo "------------------------------------------------------------------------"

    python -m src.scripts.evaluation.evaluate_upscale_quality \
        --upscaled-dir "$set_dir" \
        --original-dir "$ORIGINAL_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --quiet

    metrics_file="${OUTPUT_DIR}/${method_name}_metrics.json"
    if [ -f "$metrics_file" ]; then
        rel="${metrics_file#$PROJECT_ROOT/}"
        echo "  ✓ Metrics saved to ${rel}"
    else
        echo "  ⚠ Expected metrics file not found for ${method_name}"
    fi
done

echo ""
echo "========================================================================"
echo "✓ UPSCALE EVALUATION COMPLETE"
echo "========================================================================"
echo ""
