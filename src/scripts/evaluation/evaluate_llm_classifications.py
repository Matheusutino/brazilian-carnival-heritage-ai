"""Evaluate classification outputs produced by multimodal LLM pipelines.

This utility converts raw JSON responses into the NPZ format expected by the
standard classification evaluation module and reports multilabel metrics.

Example
-------
    python -m src.scripts.evaluation.evaluate_llm_classifications \
        --split dataset/splits/split_latest.json \
        --json results/llm/mistralai_mistral-small-3.1-24b-instruct-2503/classifications_20240101_120000.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.data.dataset_loader import MultilabelDataset
from src.scripts.evaluation.evaluate_classifications import (
    convert_to_dict_format,
    print_metrics,
)
from src.core.evaluation.classification_metrics import (
    calculate_multilabel_metrics,
    calculate_per_image_metrics,
)


def normalize_path(path_str: str) -> str:
    """Return a normalized absolute string representation for matching."""
    return str(Path(path_str).expanduser().resolve())


def parse_binary_value(value) -> int:
    """Convert mixed cell formats into binary predictions."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if value >= 1 else 0
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"1", "true", "yes", "sim", "present"}:
            return 1
    return 0


def convert_json_to_arrays(
    json_path: Path,
    dataset: MultilabelDataset,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Convert raw LLM JSON outputs to matrices aligned with dataset split."""
    data = json.loads(json_path.read_text(encoding="utf-8"))

    categories = dataset.get_categories()
    test_paths, y_true = dataset.get_test_data()
    n_samples, n_categories = y_true.shape
    y_pred = np.zeros((n_samples, n_categories), dtype=int)

    lookup: Dict[str, dict] = {}
    for item in data:
        path = item.get("image_path") or item.get("path") or item.get("image")
        if not path:
            continue
        normalized = normalize_path(path)
        lookup[normalized] = item
        lookup[Path(path).as_posix()] = item
        lookup[Path(path).name] = item

    missing: List[str] = []

    for idx, sample_path in enumerate(test_paths):
        normalized = normalize_path(sample_path)
        item = (
            lookup.get(normalized)
            or lookup.get(Path(sample_path).as_posix())
            or lookup.get(Path(sample_path).name)
        )
        if not item or item.get("status") != "success":
            missing.append(sample_path)
            continue

        classification = item.get("classification") or item.get("classificacao_categorias")
        if not classification:
            missing.append(sample_path)
            continue

        cat_block = classification.get("classificacao_categorias") if isinstance(classification, dict) else classification
        if not isinstance(cat_block, dict):
            missing.append(sample_path)
            continue

        for col, category in enumerate(categories):
            entry = cat_block.get(category)
            if isinstance(entry, dict):
                value = entry.get("valor", entry.get("value"))
            else:
                value = entry
            y_pred[idx, col] = parse_binary_value(value)

    return y_true, y_pred, categories, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM classification outputs saved as JSON."
    )

    parser.add_argument("--split", required=True, help="Train/test split JSON file.")
    parser.add_argument("--json", required=True, help="LLM classification JSON output.")
    parser.add_argument(
        "--output",
        help="Metrics JSON output path (defaults to metrics_latest.json alongside input).",
    )
    parser.add_argument(
        "--npz-out",
        help="Optional path to save converted predictions NPZ (default: same directory).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    json_path = Path(args.json).resolve()
    dataset = MultilabelDataset(args.split)
    y_true, y_pred, categories, missing = convert_json_to_arrays(json_path, dataset)

    y_true_dict = convert_to_dict_format(y_true, categories)
    y_pred_dict = convert_to_dict_format(y_pred, categories)

    metrics = calculate_multilabel_metrics(y_true_dict, y_pred_dict)
    per_image = calculate_per_image_metrics(y_true_dict, y_pred_dict)

    if not args.quiet:
        print("=" * 80)
        print("LLM CLASSIFICATION EVALUATION")
        print("=" * 80)
        print(f"Predictions file: {json_path}")
        print(f"Samples evaluated: {len(y_true)}")
        if missing:
            print(f"⚠ Missing/failed predictions: {len(missing)} (treated as zeros)")
        print_metrics(metrics, per_image)

    metrics_payload = {
        "predictions_file": str(json_path),
        "n_samples": len(y_true),
        "n_categories": len(categories),
        "categories": categories,
        "missing_predictions": [str(p) for p in missing],
        "per_image_metrics": per_image,
        "metrics": metrics,
    }

    output_path = Path(args.output) if args.output else json_path.with_name("metrics_latest.json")
    output_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.quiet:
        print(f"\n✓ Metrics saved to {output_path}")

    if args.npz_out:
        npz_path = Path(args.npz_out)
    else:
        npz_path = json_path.with_name("test_predictions_llm.npz")

    np.savez(
        npz_path,
        y_true=y_true,
        y_pred=y_pred,
        categories=np.array(categories, dtype=object),
        test_paths=np.array(dataset.get_test_data()[0], dtype=object),
    )
    if not args.quiet:
        print(f"✓ Converted NPZ saved to {npz_path}")


if __name__ == "__main__":
    main()
