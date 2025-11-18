"""Evaluate caption generation outputs from classical and LLM pipelines.

This script loads raw generation JSON files, extracts the relevant caption text,
and computes similarity metrics against human-written descriptions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts.evaluation.evaluate_caption_similarity import (
    load_human_descriptions,
    evaluate_captions,
    print_metrics_summary,
)


def flatten_classical_output(json_path: Path) -> Dict[str, str]:
    """Extract filename→caption mapping from classical caption run."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    results: Dict[str, str] = {}

    for item in data:
        if item.get("status") != "success":
            continue
        caption = item.get("caption") or item.get("caption_pt") or item.get("caption_en")
        if not caption:
            continue
        filename = item.get("image_name") or Path(item.get("image_path", "")).name
        if not filename:
            continue
        results[filename] = caption

    return results


def flatten_llm_output(json_path: Path, preferred_field: Optional[str]) -> Dict[str, str]:
    """Extract filename→caption mapping from LLM caption run."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    results: Dict[str, str] = {}

    for item in data:
        if item.get("status") != "success":
            continue

        caption_data = item.get("caption")
        caption_text: Optional[str] = None

        if isinstance(caption_data, dict):
            if preferred_field and caption_data.get(preferred_field):
                caption_text = caption_data.get(preferred_field)
            else:
                for key in ("descricao_longa", "descricao_curta", "observacoes_adicionais"):
                    value = caption_data.get(key)
                    if value:
                        caption_text = value
                        break
        elif isinstance(caption_data, str):
            caption_text = caption_data

        if not caption_text:
            continue

        filename = item.get("image_name") or Path(item.get("image_path", "")).name
        if not filename:
            continue

        results[filename] = caption_text

    return results


def label_from_path(path: Path) -> str:
    """Derive a label for metrics using the immediate parent directory."""
    return path.parent.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate caption outputs against human descriptions."
    )
    parser.add_argument("--dataset", required=True, help="Dataset CSV with human descriptions.")
    parser.add_argument(
        "--classical",
        nargs="*",
        default=[],
        help="Paths to classical caption JSON outputs.",
    )
    parser.add_argument(
        "--llm",
        nargs="*",
        default=[],
        help="Paths to LLM caption JSON outputs.",
    )
    parser.add_argument(
        "--llm-preferred-field",
        default="descricao_longa",
        help="Preferred field inside LLM structured captions for evaluation.",
    )
    parser.add_argument(
        "--bert-model",
        default="neuralmind/bert-base-portuguese-cased",
        help="BERT model identifier for BERTScore.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON file to store aggregated metrics.",
    )

    return parser.parse_args()


def evaluate_outputs(
    dataset_csv: str,
    classical_files: Iterable[Path],
    llm_files: Iterable[Path],
    bert_model: str,
    llm_field: Optional[str],
) -> List[Dict]:
    """Run evaluation for provided caption files and return metrics list."""
    human_descriptions = load_human_descriptions(dataset_csv)
    results: List[Dict] = []

    for path in classical_files:
        captions = flatten_classical_output(path)
        if not captions:
            print(f"⚠ No valid captions found in {path}")
            continue
        label = label_from_path(path)
        metrics = evaluate_captions(human_descriptions, captions, label, bert_model)
        print_metrics_summary(metrics, label)
        results.append(metrics)

    for path in llm_files:
        captions = flatten_llm_output(path, llm_field)
        if not captions:
            print(f"⚠ No valid captions found in {path}")
            continue
        label = label_from_path(path)
        metrics = evaluate_captions(human_descriptions, captions, label, bert_model)
        print_metrics_summary(metrics, label)
        results.append(metrics)

    return results


def main() -> None:
    args = parse_args()
    classical_paths = [Path(p).resolve() for p in args.classical]
    llm_paths = [Path(p).resolve() for p in args.llm]

    metrics = evaluate_outputs(
        args.dataset,
        classical_paths,
        llm_paths,
        args.bert_model,
        args.llm_preferred_field,
    )

    if args.output and metrics:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n✓ Metrics saved to {output_path}")


if __name__ == "__main__":
    main()
