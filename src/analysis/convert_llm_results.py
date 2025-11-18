"""
Converte resultados de LLM para formato CSV compatível com o analisador.
"""

import json
import pandas as pd
from pathlib import Path


def convert_llm_results_to_csv(results_path: str = 'results/classification/llm',
                                output_file: str = 'results/classification/llm_metrics.csv'):
    """
    Converte os resultados JSON dos LLMs para CSV.

    Args:
        results_path: Caminho para os resultados de LLM
        output_file: Arquivo de saída CSV
    """
    results_path = Path(results_path)

    if not results_path.exists():
        print(f"❌ Diretório não encontrado: {results_path}")
        return None

    # Procurar por todos os arquivos metrics_latest.json
    metrics_files = list(results_path.glob('*/*/metrics_latest.json'))

    if not metrics_files:
        print(f"❌ Nenhum arquivo metrics_latest.json encontrado em {results_path}")
        return None

    print(f"✓ Encontrados {len(metrics_files)} arquivos de métricas LLM")

    llm_data = []

    for metrics_file in metrics_files:
        # Extrair nome do modelo do path
        model_name = metrics_file.parent.parent.name
        strategy = metrics_file.parent.name

        # Carregar métricas
        with open(metrics_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extrair métricas principais
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

    # Criar DataFrame
    df = pd.DataFrame(llm_data)

    # Ordenar por Micro_F1
    df = df.sort_values('Micro_F1', ascending=False).reset_index(drop=True)

    # Salvar CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✓ Métricas LLM salvas em: {output_path}")
    print(f"\n=== RESUMO LLM ===")
    print(df[['Model', 'Exact_Match', 'Micro_F1', 'Macro_F1', 'Jaccard']].to_string(index=False))

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Converte resultados de LLM para CSV')
    parser.add_argument('--input', type=str, default='results/classification/llm',
                        help='Diretório com resultados LLM')
    parser.add_argument('--output', type=str, default='results/classification/llm_metrics.csv',
                        help='Arquivo de saída CSV')

    args = parser.parse_args()

    convert_llm_results_to_csv(args.input, args.output)
