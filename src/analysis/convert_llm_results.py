"""
Converts LLM results to CSV format compatible with the analyzer.
"""

import json
import pandas as pd
from pathlib import Path


def convert_llm_results_to_csv(results_path: str = 'results/classification/llm',
                                output_file: str = 'results/classification/llm_metrics.csv'):
    """
    Converts LLM JSON results to CSV format.

    Args:
        results_path: Path to LLM results directory
        output_file: Output CSV file path

    Returns:
        pd.DataFrame: DataFrame containing the LLM metrics, or None if no files found
    """
    results_path = Path(results_path)

    if not results_path.exists():
        print(f"❌ Directory not found: {results_path}")
        return None

    # Search for all metrics_latest.json files
    metrics_files = list(results_path.glob('*/*/metrics_latest.json'))

    if not metrics_files:
        print(f"❌ No metrics_latest.json files found in {results_path}")
        return None

    print(f"✓ Found {len(metrics_files)} LLM metrics files")

    llm_data = []

    for metrics_file in metrics_files:
        # Extract model name from path
        model_name = metrics_file.parent.parent.name
        strategy = metrics_file.parent.name

        # Load metrics
        with open(metrics_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract main metrics
        metrics = data.get('metrics', {})
        per_image = data.get('per_image_metrics', {})

        micro_avg = metrics.get('micro_avg', {})
        macro_avg = metrics.get('macro_avg', {})
        weighted_avg = metrics.get('weighted_avg', {})

        row = {
            'Model': f"{model_name} ({strategy})",
            'Model_Name': model_name,
            'Strategy': strategy,
            'Exact_Match': per_image.get('exact_match_ratio', 0),
            'Hamming_Loss': per_image.get('hamming_loss', 0),
            'Jaccard': per_image.get('jaccard_score', 0),
            'Micro_F1': micro_avg.get('f1', 0),
            'Micro_Precision': micro_avg.get('precision', 0),
            'Micro_Recall': micro_avg.get('recall', 0),
            'Macro_F1': macro_avg.get('f1', 0),
            'Macro_Precision': macro_avg.get('precision', 0),
            'Macro_Recall': macro_avg.get('recall', 0),
            'Weighted_F1': weighted_avg.get('f1', 0),
            'Weighted_Precision': weighted_avg.get('precision', 0),
            'Weighted_Recall': weighted_avg.get('recall', 0),
            'N_Samples': data.get('n_samples', 0),
            'N_Categories': data.get('n_categories', 0),
            'Missing_Predictions': len(data.get('missing_predictions', []))
        }

        llm_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(llm_data)

    # Sort by Micro_F1
    df = df.sort_values('Micro_F1', ascending=False).reset_index(drop=True)

    # Save CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✓ LLM metrics saved to: {output_path}")
    print(f"\n=== LLM SUMMARY ===")
    print(df[['Model', 'Exact_Match', 'Micro_F1', 'Macro_F1', 'Jaccard']].to_string(index=False))

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Converts LLM results to CSV format')
    parser.add_argument('--input', type=str, default='results/classification/llm',
                        help='Directory containing LLM results')
    parser.add_argument('--output', type=str, default='results/classification/llm_metrics.csv',
                        help='Output CSV file path')

    args = parser.parse_args()

    convert_llm_results_to_csv(args.input, args.output)
