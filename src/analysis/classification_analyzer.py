"""
Classification Analysis Script
Complete analysis of classification results: Supervised Classifiers vs LLMs
"""

import json
import math
import random
import re
import warnings
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

# Visualization configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ClassificationAnalyzer:
    """Classification results analyzer"""

    def __init__(
        self,
        results_path: str = 'results/classification',
        dataset_path: str = 'dataset',
    ):
        self.results_path = Path(results_path)
        self.dataset_path = Path(dataset_path)
        self.output_path = self.results_path / 'analysis'
        self.output_path.mkdir(exist_ok=True, parents=True)

        # Loaded data
        self.df_supervised: Optional[pd.DataFrame] = None
        self.df_llm: Optional[pd.DataFrame] = None
        self.df_combined: Optional[pd.DataFrame] = None
        self.comparison_summary: Optional[Dict] = None
        self.category_metrics: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None

        # Best models
        self.best_supervised: Optional[Dict] = None
        self.best_llm: Optional[Dict] = None

        self._load_data()

    def _load_data(self):
        """Load all necessary data for classification analysis.

        Loads and initializes all required data files for comprehensive classification
        analysis including:
        - Supervised classifier metrics from comparison_metrics.csv
        - LLM metrics from all model/strategy directories
        - Comparison summary from JSON file
        - Category-level metrics from category_metrics.csv
        - Test dataset ground truth from splits/test.csv

        After loading, combines dataframes and identifies best performing models.
        Progress is printed to console for each loaded component.

        Note:
            If any required files are missing, warnings are printed but execution
            continues to allow partial analysis with available data.
        """
        print("Loading classification data...")

        # Load comparison metrics (supervised classifiers)
        comparison_csv = self.results_path / 'comparison_metrics.csv'
        if comparison_csv.exists():
            self.df_supervised = pd.read_csv(comparison_csv)
            print(f"  ✓ Loaded supervised classifiers: {len(self.df_supervised)} methods")
        else:
            print(f"  ⚠ File not found: {comparison_csv}")

        # Load LLM metrics from all directories
        self._load_llm_metrics()

        # Load comparison summary
        summary_json = self.results_path / 'comparison_summary.json'
        if summary_json.exists():
            with open(summary_json, 'r') as f:
                self.comparison_summary = json.load(f)
            print(f"  ✓ Loaded comparison summary")
        else:
            print(f"  ⚠ File not found: {summary_json}")

        # Load category metrics
        category_csv = self.results_path / 'category_metrics.csv'
        if category_csv.exists():
            self.category_metrics = pd.read_csv(category_csv)
            print(f"  ✓ Loaded category metrics: {len(self.category_metrics)} entries")
        else:
            print(f"  ⚠ File not found: {category_csv}")

        # Load test dataset (ground truth)
        test_csv = self.dataset_path / 'splits' / 'test.csv'
        if test_csv.exists():
            self.df_test = pd.read_csv(test_csv)
            print(f"  ✓ Loaded test dataset: {len(self.df_test)} samples")
        else:
            print(f"  ⚠ File not found: {test_csv}")

        # Combine dataframes
        self._combine_dataframes()

        # Identify best models
        self._identify_best_models()

    def _load_llm_metrics(self):
        """Load LLM metrics from all strategy directories.

        Recursively scans the LLM results directory to find and load metrics from
        all model and strategy combinations. For each configuration, computes
        aggregate metrics including:
        - Micro-averaged metrics (precision, recall, F1)
        - Macro-averaged metrics (precision, recall, F1)
        - Weighted F1 score
        - Per-image metrics (exact match, hamming loss, Jaccard)

        The function expects the directory structure:
        results/classification/llm/{model_name}/{strategy_name}/metrics_latest.json

        Results are stored in self.df_llm as a pandas DataFrame with one row per
        model/strategy combination.

        Note:
            If any metrics file is corrupted or missing required fields, a warning
            is printed and that configuration is skipped.
        """
        llm_dir = self.results_path / 'llm'

        if not llm_dir.exists():
            print(f"  ⚠ LLM directory not found: {llm_dir}")
            return

        llm_data = []

        # Iterate through model directories
        for model_dir in llm_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            # Iterate through strategy directories
            for strategy_dir in model_dir.iterdir():
                if not strategy_dir.is_dir():
                    continue

                strategy_name = strategy_dir.name
                metrics_file = strategy_dir / 'metrics_latest.json'

                if not metrics_file.exists():
                    continue

                try:
                    with open(metrics_file, 'r') as f:
                        metrics_data = json.load(f)

                    # Calculate aggregate metrics
                    per_image = metrics_data.get('per_image_metrics', {})
                    category_metrics = metrics_data.get('metrics', {})

                    # Calculate macro F1 from per-category metrics
                    f1_scores = [cat_data.get('f1', 0) for cat_data in category_metrics.values()]
                    macro_f1 = np.mean(f1_scores) if f1_scores else 0

                    # Calculate macro precision and recall
                    precisions = [cat_data.get('precision', 0) for cat_data in category_metrics.values()]
                    recalls = [cat_data.get('recall', 0) for cat_data in category_metrics.values()]
                    macro_precision = np.mean(precisions) if precisions else 0
                    macro_recall = np.mean(recalls) if recalls else 0

                    # Calculate micro metrics (weighted by support)
                    total_tp = sum(cat_data.get('tp', 0) for cat_data in category_metrics.values())
                    total_fp = sum(cat_data.get('fp', 0) for cat_data in category_metrics.values())
                    total_fn = sum(cat_data.get('fn', 0) for cat_data in category_metrics.values())

                    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

                    # Calculate weighted F1
                    supports = [cat_data.get('support', 0) for cat_data in category_metrics.values()]
                    total_support = sum(supports)
                    weighted_f1 = sum(f1 * supp for f1, supp in zip(f1_scores, supports)) / total_support if total_support > 0 else 0

                    llm_data.append({
                        'Model': f'{model_name} ({strategy_name})',
                        'Model_Name': model_name,
                        'Strategy': strategy_name,
                        'Exact_Match': per_image.get('exact_match_ratio', 0),
                        'Hamming_Loss': per_image.get('hamming_loss', 0),
                        'Jaccard': per_image.get('jaccard_score', 0),
                        'Micro_F1': micro_f1,
                        'Micro_Precision': micro_precision,
                        'Micro_Recall': micro_recall,
                        'Macro_F1': macro_f1,
                        'Macro_Precision': macro_precision,
                        'Macro_Recall': macro_recall,
                        'Weighted_F1': weighted_f1,
                        'N_Samples': metrics_data.get('n_samples', 0),
                        'N_Categories': metrics_data.get('n_categories', 0),
                        'metrics_path': str(metrics_file),
                        'strategy_dir': str(strategy_dir)
                    })

                except Exception as e:
                    print(f"  ⚠ Error loading {metrics_file}: {e}")

        if llm_data:
            self.df_llm = pd.DataFrame(llm_data)
            print(f"  ✓ Loaded LLM results: {len(self.df_llm)} configurations")

            # Show breakdown
            n_models = self.df_llm['Model_Name'].nunique()
            n_strategies = self.df_llm['Strategy'].nunique()
            print(f"    - {n_models} models × {n_strategies} strategies")
        else:
            print("  ⚠ No LLM metrics found")

    def _combine_dataframes(self):
        """Combine supervised and LLM dataframes into a unified dataframe.

        Merges the supervised classifier metrics and LLM metrics into a single
        dataframe for unified analysis and comparison. Renames columns as needed
        to ensure consistency between the two sources (e.g., 'Classifier' vs 'Model').

        The combined dataframe enables direct comparison and visualization of
        performance across all classification methods (both traditional ML and LLM-based).

        Sets self.df_combined with the merged results, or None if either source
        dataframe is missing.

        Note:
            Prints the total number of configurations in the combined dataframe.
        """
        if self.df_supervised is None or self.df_llm is None:
            return

        # Rename columns for consistency
        df_sup = self.df_supervised.copy()
        df_sup['Model_Type'] = 'Supervised'
        df_sup['Model'] = df_sup['Method']

        df_llm = self.df_llm.copy()
        df_llm['Model_Type'] = 'LLM'
        df_llm['Model'] = df_llm['Model']

        # Select common columns
        common_cols = ['Model', 'Model_Type', 'Exact_Match', 'Hamming_Loss', 'Jaccard',
                       'Micro_F1', 'Macro_F1', 'Weighted_F1']

        df_sup_subset = df_sup[common_cols]
        df_llm_subset = df_llm[common_cols]

        self.df_combined = pd.concat([df_sup_subset, df_llm_subset], ignore_index=True)
        print(f"  ✓ Combined dataframe: {len(self.df_combined)} total configurations")

    def _identify_best_models(self):
        """Identify best supervised and LLM models by Macro F1 score.

        Determines the best performing models in each category (supervised classifiers
        and LLM-based classifiers) based on macro-averaged F1 score. Macro F1 is used
        as it provides balanced performance assessment across all categories regardless
        of class imbalance.

        Sets:
            self.best_supervised (dict): Row from df_supervised with highest Macro_F1
            self.best_llm (dict): Row from df_llm with highest Macro_F1

        The best models are used for detailed analysis and comparison throughout
        the analyzer.

        Note:
            Prints information about the identified best models to console.
        """
        if self.df_supervised is not None:
            best_idx = self.df_supervised['Macro_F1'].idxmax()
            self.best_supervised = {
                'method': self.df_supervised.loc[best_idx, 'Method'],
                'feature_model': self.df_supervised.loc[best_idx, 'Feature_Model'],
                'classifier': self.df_supervised.loc[best_idx, 'Classifier'],
                'macro_f1': self.df_supervised.loc[best_idx, 'Macro_F1'],
                'index': best_idx
            }
            print(f"\n  🏆 Best Supervised: {self.best_supervised['method']} "
                  f"(Macro F1: {self.best_supervised['macro_f1']:.4f})")

        if self.df_llm is not None:
            best_idx = self.df_llm['Macro_F1'].idxmax()
            self.best_llm = {
                'model': self.df_llm.loc[best_idx, 'Model'],
                'model_name': self.df_llm.loc[best_idx, 'Model_Name'],
                'strategy': self.df_llm.loc[best_idx, 'Strategy'],
                'macro_f1': self.df_llm.loc[best_idx, 'Macro_F1'],
                'index': best_idx,
                'metrics_path': self.df_llm.loc[best_idx, 'metrics_path'],
                'strategy_dir': self.df_llm.loc[best_idx, 'strategy_dir']
            }
            print(f"  🏆 Best LLM: {self.best_llm['model_name']} ({self.best_llm['strategy']}) "
                  f"(Macro F1: {self.best_llm['macro_f1']:.4f})")

    @staticmethod
    def _normalize_category_label(label: Optional[str]) -> str:
        if not label:
            return ""
        normalized = str(label).replace("\\n", " ").replace("\n", " ")
        normalized = re.sub(r"\s+", " ", normalized).strip().lower()
        normalized = normalized.replace(" /", "/").replace("/ ", "/")
        return normalized

    def _resolve_image_path(self, image_path: str) -> Optional[Path]:
        """Return absolute Path for an image, if it exists."""
        if not image_path:
            return None

        candidates = []
        path = Path(image_path)
        candidates.append(path)

        if not path.is_absolute():
            candidates.append(Path.cwd() / path)
            candidates.append(self.dataset_path / path.name)

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        return None

    def _load_best_supervised_predictions(self) -> Optional[Dict[str, np.ndarray]]:
        if self.best_supervised is None:
            print("  ⚠ Best supervised model not identified")
            return None

        predictions_path = (
            self.results_path / 'supervised_classifiers' /
            self.best_supervised['feature_model'] /
            self.best_supervised['classifier'] /
            'test_predictions.npz'
        )

        if not predictions_path.exists():
            print(f"  ⚠ Predictions file not found for supervised model: {predictions_path}")
            return None

        data = np.load(predictions_path, allow_pickle=True)
        return {
            'paths': data['test_paths'].tolist(),
            'y_true': data['y_true'].astype(int),
            'y_pred': data['y_pred'].astype(int),
            'categories': data['categories'].tolist(),
        }

    def _load_best_llm_predictions(self, target_categories: List[str]) -> Optional[Dict[str, np.ndarray]]:
        if self.best_llm is None:
            print("  ⚠ Best LLM model not identified")
            return None

        strategy_dir = Path(self.best_llm.get('strategy_dir', ''))
        if not strategy_dir.exists():
            print(f"  ⚠ LLM strategy directory not found: {strategy_dir}")
            return None

        classification_files = sorted(strategy_dir.glob('classifications_*.json'))
        if not classification_files:
            print(f"  ⚠ No classification files found in {strategy_dir}")
            return None

        metrics_file = strategy_dir / 'metrics_latest.json'
        if not metrics_file.exists():
            print(f"  ⚠ Metrics file not found for LLM strategy: {metrics_file}")
            return None

        with open(classification_files[-1], 'r') as f:
            classifications = json.load(f)

        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)

        target_index = {
            self._normalize_category_label(cat): idx for idx, cat in enumerate(target_categories)
        }

        llm_predictions: Dict[str, np.ndarray] = {}
        for item in classifications:
            if item.get('status') != 'success':
                continue

            image_path = item.get('image_path') or item.get('metadata', {}).get('local_path')
            if not image_path:
                continue

            classification = item.get('classification', {}).get('classificacao_categorias', {})
            pred_vector = np.zeros(len(target_categories), dtype=int)

            for label, info in classification.items():
                idx = target_index.get(self._normalize_category_label(label))
                if idx is None:
                    continue
                value = info.get('valor')
                pred_vector[idx] = 1 if value == 1 else 0

            llm_predictions[image_path] = pred_vector

        if not llm_predictions:
            print("  ⚠ LLM predictions could not be parsed")
            return None

        return {'predictions': llm_predictions, 'categories': target_categories}

    def _save_figure(self, fig, filename_base: str, dpi: int = 300):
        """Save figure in PNG and PDF."""
        png_path = self.output_path / f'{filename_base}.png'
        pdf_path = self.output_path / f'{filename_base}.pdf'
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
        fig.savefig(pdf_path, dpi=dpi, bbox_inches='tight')

    def analyze_embeddings_by_classifier(self):
        """1. Analysis: Comparison of embeddings by classifier"""
        print("\n" + "="*80)
        print("1. ANALYSIS: EMBEDDINGS COMPARISON BY CLASSIFIER")
        print("="*80)

        if self.df_supervised is None:
            print("  ⚠ No supervised data available")
            return

        # Mapping for embedding names
        embedding_name_mapping = {
            'dinov2-base': 'DINOv2 Base',
            'dinov2-small': 'DINOv2 Small',
            'convnextv2-tiny': 'ConvNeXt V2 Tiny'
        }

        # Mapping for classifier names
        classifier_name_mapping = {
            'lda': 'LDA',
            'nb': 'Naive Bayes',
            'passive_aggressive': 'Passive Aggressive',
            'rf': 'Random Forest',
            'tree': 'Decision Tree',
            'xgb': 'XGBoost'
        }

        # Get unique embeddings and classifiers
        embeddings = sorted(self.df_supervised['Feature_Model'].unique())
        classifiers = sorted(self.df_supervised['Classifier'].unique())

        print(f"\n  📊 Embeddings: {', '.join(embeddings)}")
        print(f"  📊 Classifiers: {', '.join(classifiers)}")

        # Create subplots for each metric
        metrics = ['Macro_F1', 'Micro_F1', 'Exact_Match', 'Jaccard']
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            # Prepare data for grouped bar chart
            x = np.arange(len(classifiers))

            colors = plt.cm.Set3(np.linspace(0, 1, len(embeddings)))

            # Espaço total ocupado pelas barras dentro de cada grupo (0–1)
            group_width = 0.8          # deixa ~20% de gap entre grupos
            n = len(embeddings)
            width = group_width / n     # largura de cada barra

            for i, embedding in enumerate(embeddings):
                values = []
                for classifier in classifiers:
                    row = self.df_supervised[
                        (self.df_supervised['Feature_Model'] == embedding) &
                        (self.df_supervised['Classifier'] == classifier)
                    ]
                    values.append(row[metric].iloc[0] if len(row) > 0 else 0)

                # centraliza o bloco e posiciona cada barra dentro do bloco
                offset = (-group_width / 2) + i * width + (width / 2)
                embedding_label = embedding_name_mapping.get(embedding, embedding)
                bars = ax.bar(x + offset, values, width, label=embedding_label, color=colors[i], alpha=0.8)

                # rótulos um pouco acima para não encostar
                for bar, value in zip(bars, values):
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{value:.3f}',
                            ha='center', va='bottom', fontsize=7)



            ax.set_xlabel('Classifier', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' '), fontsize=12, fontweight='bold')
            ax.set_title(f'{metric.replace("_", " ")} by Classifier and Embedding',
                        fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            classifier_labels = [classifier_name_mapping.get(c, c) for c in classifiers]
            ax.set_xticklabels(classifier_labels, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)

        # Create a single legend below all subplots
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(embeddings),
                  fontsize=10, bbox_to_anchor=(0.5, -0.02))

        fig.suptitle('Embeddings Comparison Across Classifiers',
                     fontsize=16, fontweight='bold', y=0.98)
        fig.tight_layout()
        self._save_figure(fig, '01_embeddings_by_classifier', dpi=300)
        plt.close(fig)

        print("  ✓ Charts saved: 01_embeddings_by_classifier.png/.pdf")

        # Statistical summary
        print("\n  📈 Best Embedding per Classifier (by Macro F1):\n")
        for classifier in classifiers:
            subset = self.df_supervised[self.df_supervised['Classifier'] == classifier]
            best_idx = subset['Macro_F1'].idxmax()
            best_embedding = subset.loc[best_idx, 'Feature_Model']
            best_f1 = subset.loc[best_idx, 'Macro_F1']
            print(f"    {classifier:20s}: {best_embedding:20s} (F1: {best_f1:.4f})")

    def analyze_llm_comparison(self):
        """2. Analysis: Comparison between LLM methods"""
        print("\n" + "="*80)
        print("2. ANALYSIS: LLM METHODS COMPARISON")
        print("="*80)

        if self.df_llm is None:
            print("  ⚠ No LLM data available")
            return

        # Get unique models and strategies
        models = sorted(self.df_llm['Model_Name'].unique())
        strategies = sorted(self.df_llm['Strategy'].unique())

        print(f"\n  📊 Models: {len(models)}")
        for model in models:
            print(f"    - {model}")
        print(f"  📊 Strategies: {', '.join(strategies)}")

        # Create comparison chart
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()

        metrics = ['Macro_F1', 'Micro_F1', 'Exact_Match', 'Jaccard']

        # Strategy name mapping
        strategy_mapping = {
            'basic': 'Basic',
            'with_descriptions': 'Basic With Description',
            'cot': 'CoT',
            'dual_role': 'Dual Role'
        }

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            # Prepare data
            data_for_plot = []
            labels = []

            for model in models:
                for strategy in strategies:
                    row = self.df_llm[
                        (self.df_llm['Model_Name'] == model) &
                        (self.df_llm['Strategy'] == strategy)
                    ]
                    if len(row) > 0:
                        data_for_plot.append(row[metric].iloc[0])
                        # Format model name
                        if 'llama' in model.lower():
                            if 'scout' in model.lower():
                                model_short = 'Llama 4 Scout'
                            elif 'maverick' in model.lower():
                                model_short = 'Llama 4 Maverick'
                            else:
                                model_short = 'Llama 4'
                        else:
                            model_short = model.split('_')[-1].split('-')[0].capitalize()

                        # Format strategy name
                        strategy_clean = strategy.replace('zero_shot_', '')
                        strategy_short = strategy_mapping.get(strategy_clean, strategy_clean)
                        labels.append(f"{model_short}\n{strategy_short}")

            x = np.arange(len(labels))
            colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))

            bars = ax.bar(x, data_for_plot, color=colors, alpha=0.8)
            ax.set_xlabel('Model + Strategy', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' '), fontsize=12, fontweight='bold')
            ax.set_title(f'{metric.replace("_", " ")} Comparison', fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, data_for_plot):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=7)

        fig.suptitle('LLM Methods Comparison', fontsize=16, fontweight='bold', y=1.00)
        fig.tight_layout()
        self._save_figure(fig, '02_llm_comparison', dpi=300)
        plt.close(fig)

        print("  ✓ Charts saved: 02_llm_comparison.png/.pdf")

        # Best per model
        print("\n  🏆 Best Strategy per Model (by Macro F1):\n")
        for model in models:
            subset = self.df_llm[self.df_llm['Model_Name'] == model]
            best_idx = subset['Macro_F1'].idxmax()
            best_strategy = subset.loc[best_idx, 'Strategy']
            best_f1 = subset.loc[best_idx, 'Macro_F1']
            model_short = model.split('/')[-1] if '/' in model else model
            print(f"    {model_short:50s}: {best_strategy:30s} (F1: {best_f1:.4f})")

    def analyze_supervised_vs_llm(self):
        """3. Analysis: Supervised vs LLM comparison"""
        print("\n" + "="*80)
        print("3. ANALYSIS: SUPERVISED VS LLM")
        print("="*80)

        if self.df_combined is None:
            print("  ⚠ No combined data available")
            return

        # Box plot comparison
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()

        metrics = ['Macro_F1', 'Micro_F1', 'Exact_Match', 'Jaccard', 'Weighted_F1', 'Hamming_Loss']

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            data_supervised = self.df_combined[
                self.df_combined['Model_Type'] == 'Supervised'
            ][metric].dropna()

            data_llm = self.df_combined[
                self.df_combined['Model_Type'] == 'LLM'
            ][metric].dropna()

            bp = ax.boxplot(
                [data_supervised, data_llm],
                labels=['Supervised', 'LLM'],
                patch_artist=True,
                showmeans=False
            )

            colors = ['#3498db', '#e74c3c']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_ylabel(metric.replace('_', ' '), fontsize=11, fontweight='bold')
            ax.set_title(metric.replace('_', ' '), fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

        fig.suptitle('Supervised Classifiers vs LLMs - Metrics Comparison',
                     fontsize=16, fontweight='bold', y=1.00)
        fig.tight_layout()
        self._save_figure(fig, '03_supervised_vs_llm', dpi=300)
        plt.close(fig)

        print("  ✓ Charts saved: 03_supervised_vs_llm.png/.pdf")

        # Statistical summary
        print("\n  📊 Statistical Summary:\n")
        print("  " + "-"*76)
        print(f"  {'Metric':<20s} {'Supervised Mean':<20s} {'LLM Mean':<20s} {'Winner':<15s}")
        print("  " + "-"*76)

        for metric in ['Macro_F1', 'Micro_F1', 'Exact_Match', 'Jaccard', 'Weighted_F1']:
            sup_mean = self.df_combined[self.df_combined['Model_Type'] == 'Supervised'][metric].mean()
            llm_mean = self.df_combined[self.df_combined['Model_Type'] == 'LLM'][metric].mean()
            winner = 'Supervised' if sup_mean > llm_mean else 'LLM'
            print(f"  {metric:<20s} {sup_mean:<20.4f} {llm_mean:<20.4f} {winner:<15s}")

        print("  " + "-"*76)

    def analyze_category_difficulty(self):
        """4. Analysis: Top 5 easiest and hardest categories"""
        print("\n" + "="*80)
        print("4. ANALYSIS: CATEGORY DIFFICULTY")
        print("="*80)

        # Load category performance for best supervised model
        if self.comparison_summary is None:
            print("  ⚠ No comparison summary available for supervised")
            return

        category_perf_sup = self.comparison_summary.get('category_performance', {})

        if not category_perf_sup:
            print("  ⚠ No category performance data for supervised")
            return

        # Load category performance for best LLM model
        category_perf_llm = None
        if self.best_llm:
            metrics_path = Path(self.best_llm.get('metrics_path', ''))
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    llm_metrics = json.load(f)
                    category_perf_llm = llm_metrics.get('metrics', {})

        # Convert to DataFrames
        cat_df_sup = pd.DataFrame.from_dict(category_perf_sup, orient='index')
        cat_df_sup = cat_df_sup.reset_index().rename(columns={'index': 'Category'})
        cat_df_sup = cat_df_sup.sort_values('F1', ascending=False)

        # Top 5 easiest and hardest (supervised)
        top_5_easy_sup = cat_df_sup.head(5)
        top_5_hard_sup = cat_df_sup.tail(5)

        print("\n  ✅ Top 5 EASIEST Categories (Best Supervised Model):\n")
        for idx, row in top_5_easy_sup.iterrows():
            print(f"    {row['Category']:30s} - F1: {row['F1']:.4f}  "
                  f"(Precision: {row['Precision']:.4f}, Recall: {row['Recall']:.4f}, Support: {row['Support']:.0f})")

        print("\n  ❌ Top 5 HARDEST Categories (Best Supervised Model):\n")
        for idx, row in top_5_hard_sup.iterrows():
            print(f"    {row['Category']:30s} - F1: {row['F1']:.4f}  "
                  f"(Precision: {row['Precision']:.4f}, Recall: {row['Recall']:.4f}, Support: {row['Support']:.0f})")

        # Process LLM categories if available
        cat_df_llm = None
        if category_perf_llm:
            cat_data_llm = []
            for cat_name, metrics in category_perf_llm.items():
                cat_data_llm.append({
                    'Category': cat_name,
                    'F1': metrics.get('f1', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'Support': metrics.get('support', 0)
                })
            cat_df_llm = pd.DataFrame(cat_data_llm).sort_values('F1', ascending=False)

            top_5_easy_llm = cat_df_llm.head(5)
            top_5_hard_llm = cat_df_llm.tail(5)

            print("\n  ✅ Top 5 EASIEST Categories (Best LLM Model):\n")
            for idx, row in top_5_easy_llm.iterrows():
                print(f"    {row['Category']:30s} - F1: {row['F1']:.4f}  "
                      f"(Precision: {row['Precision']:.4f}, Recall: {row['Recall']:.4f}, Support: {row['Support']:.0f})")

            print("\n  ❌ Top 5 HARDEST Categories (Best LLM Model):\n")
            for idx, row in top_5_hard_llm.iterrows():
                print(f"    {row['Category']:30s} - F1: {row['F1']:.4f}  "
                      f"(Precision: {row['Precision']:.4f}, Recall: {row['Recall']:.4f}, Support: {row['Support']:.0f})")

        # Visualization
        n_plots = 4 if cat_df_llm is not None else 2
        fig, axes = plt.subplots(2, 2, figsize=(20, 14)) if cat_df_llm is not None else plt.subplots(1, 2, figsize=(20, 8))

        if cat_df_llm is not None:
            axes = axes.flatten()
        else:
            axes = list(axes)

        # define limite global para x
        max_f1s = [top_5_easy_sup['F1'].max(), top_5_hard_sup['F1'].max()]
        if cat_df_llm is not None:
            max_f1s += [top_5_easy_llm['F1'].max(), top_5_hard_llm['F1'].max()]
        xmax_all = max(max_f1s) * 1.10  # 10% folga


        # ---- SUPERISED - EASIEST ----
        ax = axes[0]
        y_pos = np.arange(len(top_5_easy_sup))
        bars = ax.barh(y_pos, top_5_easy_sup['F1'], color='#2ecc71', alpha=0.8)
        ax.set_xlim(0, xmax_all)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_5_easy_sup['Category'])
        ax.invert_yaxis()
        ax.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('Top 5 Easiest Categories\n(Best Supervised Model)', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.bar_label(bars, fmt='%.3f', padding=3, label_type='edge')


        # ---- SUPERVISED - HARDEST ----
        ax = axes[1]
        y_pos = np.arange(len(top_5_hard_sup))
        bars = ax.barh(y_pos, top_5_hard_sup['F1'], color='#e74c3c', alpha=0.8)
        ax.set_xlim(0, xmax_all)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_5_hard_sup['Category'])
        ax.invert_yaxis()
        ax.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('Top 5 Hardest Categories\n(Best Supervised Model)', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.bar_label(bars, fmt='%.3f', padding=3, label_type='edge')


        # ---- LLM PLOTS ----
        if cat_df_llm is not None:
            # LLM - Easiest
            ax = axes[2]
            y_pos = np.arange(len(top_5_easy_llm))
            bars = ax.barh(y_pos, top_5_easy_llm['F1'], color='#3498db', alpha=0.8)
            ax.set_xlim(0, xmax_all)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_5_easy_llm['Category'])
            ax.invert_yaxis()
            ax.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
            ax.set_title('Top 5 Easiest Categories\n(Best LLM Model)', fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            ax.bar_label(bars, fmt='%.3f', padding=3, label_type='edge')

            # LLM - Hardest
            ax = axes[3]
            y_pos = np.arange(len(top_5_hard_llm))
            bars = ax.barh(y_pos, top_5_hard_llm['F1'], color='#f39c12', alpha=0.8)
            ax.set_xlim(0, xmax_all)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_5_hard_llm['Category'])
            ax.invert_yaxis()
            ax.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
            ax.set_title('Top 5 Hardest Categories\n(Best LLM Model)', fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            ax.bar_label(bars, fmt='%.3f', padding=3, label_type='edge')


        fig.suptitle('Category Difficulty Analysis - Best Models', fontsize=16, fontweight='bold')
        fig.tight_layout()
        self._save_figure(fig, '04_category_difficulty', dpi=300)
        plt.close(fig)

        print("\n  ✓ Charts saved: 04_category_difficulty.png/.pdf")


    def analyze_prompts_by_llm(self):
        """5. Analysis: Prompt comparison for each LLM"""
        print("\n" + "="*80)
        print("5. ANALYSIS: PROMPT STRATEGIES BY LLM")
        print("="*80)

        if self.df_llm is None:
            print("  ⚠ No LLM data available")
            return

        # Strategy name mapping
        strategy_mapping = {
            'basic': 'Basic',
            'with_descriptions': 'Basic With Description',
            'cot': 'CoT',
            'dual_role': 'Dual Role'
        }

        # Define desired order for strategies (with zero_shot_ prefix)
        strategy_order = ['zero_shot_basic', 'zero_shot_with_descriptions', 'zero_shot_cot', 'zero_shot_dual_role']

        models = sorted(self.df_llm['Model_Name'].unique())
        all_strategies = self.df_llm['Strategy'].unique()

        # Order strategies according to strategy_order
        strategies = [s for s in strategy_order if s in all_strategies]

        # Create radar chart for each model
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 7),
                                subplot_kw={'projection': 'polar'})

        if n_models == 1:
            axes = [axes]

        metrics = ['Macro_F1', 'Micro_F1', 'Exact_Match', 'Jaccard', 'Weighted_F1']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))

        for ax, model in zip(axes, models):
            for strategy, color in zip(strategies, colors):
                row = self.df_llm[
                    (self.df_llm['Model_Name'] == model) &
                    (self.df_llm['Strategy'] == strategy)
                ]

                if len(row) == 0:
                    continue

                values = [row[m].iloc[0] for m in metrics]
                values += values[:1]

                strategy_clean = strategy.replace('zero_shot_', '')
                strategy_label = strategy_mapping.get(strategy_clean, strategy_clean)
                ax.plot(angles, values, 'o-', linewidth=2, label=strategy_label, color=color)
                ax.fill(angles, values, alpha=0.15, color=color)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m.replace('_', '\n') for m in metrics], fontsize=9)
            ax.set_ylim(0, 1)
            # Format model name
            if 'llama' in model.lower():
                if 'scout' in model.lower():
                    model_short = 'Llama Scout'
                elif 'maverick' in model.lower():
                    model_short = 'Llama Maverick'
                else:
                    model_short = 'Llama'
            else:
                model_short = model.split('_')[-1].capitalize()
            ax.set_title(model_short, fontsize=13, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle('Prompt Strategy Comparison by LLM Model (Radar)',
                     fontsize=16, fontweight='bold')
        fig.tight_layout()
        self._save_figure(fig, '05a_prompts_by_llm_radar', dpi=300)
        plt.close(fig)

        print("  ✓ Charts saved: 05a_prompts_by_llm_radar.png/.pdf")

        # Create bar chart comparison
        fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 8))

        if n_models == 1:
            axes = [axes]

        for ax, model in zip(axes, models):
            model_data = self.df_llm[self.df_llm['Model_Name'] == model]

            x = np.arange(len(strategies))
            width = 0.15

            metrics_to_plot = ['Macro_F1', 'Micro_F1', 'Exact_Match', 'Jaccard', 'Weighted_F1']
            colors_bar = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

            for i, metric in enumerate(metrics_to_plot):
                values = []
                for strategy in strategies:
                    row = model_data[model_data['Strategy'] == strategy]
                    values.append(row[metric].iloc[0] if len(row) > 0 else 0)

                offset = (i - len(metrics_to_plot)/2) * width + width/2
                bars = ax.bar(x + offset, values, width, label=metric.replace('_', ' '), color=colors_bar[i], alpha=0.8)

                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=6, rotation=0)

            ax.set_xlabel('Strategy', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            # Format model name
            if 'llama' in model.lower():
                if 'scout' in model.lower():
                    model_short = 'Llama Scout'
                elif 'maverick' in model.lower():
                    model_short = 'Llama Maverick'
                else:
                    model_short = 'Llama'
            else:
                model_short = model.split('_')[-1].capitalize()
            ax.set_title(f'{model_short}', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            strategy_labels = [strategy_mapping.get(s.replace('zero_shot_', ''), s.replace('zero_shot_', '')) for s in strategies]
            ax.set_xticklabels(strategy_labels, rotation=45, ha='right')
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1.1])

        fig.suptitle('Prompt Strategy Comparison by LLM Model (Bar Charts)',
                     fontsize=16, fontweight='bold')
        fig.tight_layout()
        self._save_figure(fig, '05b_prompts_by_llm_bars', dpi=300)
        plt.close(fig)

        print("  ✓ Charts saved: 05b_prompts_by_llm_bars.png/.pdf")

        # Detailed comparison table
        print("\n  📊 Detailed Comparison (Macro F1):\n")
        pivot_table = self.df_llm.pivot_table(
            values='Macro_F1',
            index='Strategy',
            columns='Model_Name',
            aggfunc='first'
        )
        print(pivot_table.to_string())

        # ===== NEW: Aggregate prompt comparison across all models =====
        print("\n  📊 Creating aggregate prompt comparison chart...")

        # Group by strategy and calculate mean across all models
        strategy_stats = self.df_llm.groupby('Strategy').agg({
            'Macro_F1': ['mean', 'std'],
            'Micro_F1': ['mean', 'std'],
            'Exact_Match': ['mean', 'std'],
            'Jaccard': ['mean', 'std'],
            'Weighted_F1': ['mean', 'std']
        }).reset_index()

        # Flatten column names
        strategy_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                                  for col in strategy_stats.columns.values]

        # Order strategies
        strategy_stats['Strategy'] = pd.Categorical(
            strategy_stats['Strategy'],
            categories=strategies,
            ordered=True
        )
        strategy_stats = strategy_stats.sort_values('Strategy')

        # Create bar chart with error bars
        # X-axis: metrics, Legend: prompts
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        # Order: Micro F1, Weighted F1, Macro F1, Exact Match, Jaccard
        metrics_to_plot = ['Micro_F1', 'Weighted_F1', 'Macro_F1', 'Exact_Match', 'Jaccard']
        metric_labels = ['Micro F1', 'Weighted F1', 'Macro F1', 'Exact Match', 'Jaccard']

        x = np.arange(len(metrics_to_plot))
        width = 0.2
        colors_bar = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

        for i, strategy in enumerate(strategies):
            means = []
            stds = []
            row = strategy_stats[strategy_stats['Strategy'] == strategy]

            if len(row) > 0:
                for metric in metrics_to_plot:
                    means.append(row[f'{metric}_mean'].iloc[0])
                    stds.append(row[f'{metric}_std'].iloc[0])
            else:
                means = [0] * len(metrics_to_plot)
                stds = [0] * len(metrics_to_plot)

            offset = (i - len(strategies)/2) * width + width/2
            strategy_label = strategy_mapping.get(strategy.replace('zero_shot_', ''),
                                                 strategy.replace('zero_shot_', ''))
            bars = ax.bar(x + offset, means, width, yerr=stds,
                         label=strategy_label,
                         color=colors_bar[i], alpha=0.8,
                         capsize=3)

            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=7, rotation=0)

        ax.set_xlabel('Metric', fontsize=13, fontweight='bold')
        ax.set_ylabel('Score', fontsize=13, fontweight='bold')
        ax.set_title('Prompt Strategy Comparison (Aggregated Across All Models)',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, rotation=0, ha='center')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 0.7])

        # Place legend below the chart, horizontally
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
                 ncol=len(strategies), fontsize=10, frameon=True)

        fig.tight_layout()
        self._save_figure(fig, '05c_prompts_aggregate_comparison', dpi=300)
        plt.close(fig)

        print("  ✓ Charts saved: 05c_prompts_aggregate_comparison.png/.pdf")

        # Print aggregate statistics
        print("\n  📊 Aggregate Statistics by Prompt (Mean ± Std):\n")
        print("  " + "-"*90)
        print(f"  {'Strategy':<30s} {'Macro F1':<20s} {'Micro F1':<20s} {'Exact Match':<20s}")
        print("  " + "-"*90)
        for strategy in strategies:
            row = strategy_stats[strategy_stats['Strategy'] == strategy]
            if len(row) > 0:
                strategy_label = strategy_mapping.get(strategy.replace('zero_shot_', ''),
                                                     strategy.replace('zero_shot_', ''))
                macro_f1 = f"{row['Macro_F1_mean'].iloc[0]:.4f} ± {row['Macro_F1_std'].iloc[0]:.4f}"
                micro_f1 = f"{row['Micro_F1_mean'].iloc[0]:.4f} ± {row['Micro_F1_std'].iloc[0]:.4f}"
                exact_match = f"{row['Exact_Match_mean'].iloc[0]:.4f} ± {row['Exact_Match_std'].iloc[0]:.4f}"
                print(f"  {strategy_label:<30s} {macro_f1:<20s} {micro_f1:<20s} {exact_match:<20s}")
        print("  " + "-"*90)

    def analyze_confusion_matrices(self):
        """6. Analysis: Confusion matrices for best models"""
        print("\n" + "="*80)
        print("6. ANALYSIS: CONFUSION MATRICES")
        print("="*80)

        # ========== SUPERVISED MODEL ==========
        if self.best_supervised is None:
            print("  ⚠ Best supervised model not identified")
        else:
            # Load predictions for best supervised model
            supervised_path = (
                self.results_path / 'supervised_classifiers' /
                self.best_supervised['feature_model'] /
                self.best_supervised['classifier'] /
                'test_predictions.npz'
            )

            if not supervised_path.exists():
                print(f"  ⚠ Predictions file not found: {supervised_path}")
            else:
                print(f"\n  📂 Loading predictions (Supervised): {supervised_path.name}")

                data = np.load(supervised_path, allow_pickle=True)
                y_true = data['y_true']
                y_pred = data['y_pred']

                # Try different keys for label names
                label_names = None
                if 'label_names' in data:
                    label_names = data['label_names'].tolist()
                elif 'categories' in data:
                    label_names = data['categories'].tolist()

                if label_names is None:
                    print("  ⚠ Label names not found in predictions file")
                else:
                    print(f"  ✓ Loaded {len(y_true)} predictions with {len(label_names)} categories")

                    # Since this is multi-label, we need to handle it differently
                    # Calculate confusion for each label
                    n_labels = len(label_names)

                    # Create figure with subplots
                    n_rows = int(np.ceil(np.sqrt(n_labels)))
                    n_cols = int(np.ceil(n_labels / n_rows))

                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
                    axes = axes.flatten() if n_labels > 1 else [axes]

                    for idx, label_name in enumerate(label_names[:len(axes)]):
                        ax = axes[idx]

                        # Binary confusion matrix for this label
                        y_true_label = y_true[:, idx] if y_true.ndim > 1 else (y_true == idx).astype(int)
                        y_pred_label = y_pred[:, idx] if y_pred.ndim > 1 else (y_pred == idx).astype(int)

                        cm = confusion_matrix(y_true_label, y_pred_label)

                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                   xticklabels=['Negative', 'Positive'],
                                   yticklabels=['Negative', 'Positive'],
                                   cbar=False)

                        ax.set_title(label_name, fontsize=10, fontweight='bold')
                        ax.set_xlabel('Predicted', fontsize=9)
                        ax.set_ylabel('True', fontsize=9)

                    # Hide extra subplots
                    for idx in range(n_labels, len(axes)):
                        axes[idx].axis('off')

                    # Format supervised model name
                    embedding_mapping = {
                        'dinov2-base': 'DINOv2 Base',
                        'dinov2-small': 'DINOv2 Small',
                        'convnextv2-tiny': 'ConvNeXt V2 Tiny',
                        'convnextv2-base': 'ConvNeXt V2 Base'
                    }
                    classifier_mapping = {
                        'lda': 'LDA',
                        'nb': 'Naive Bayes',
                        'passive_aggressive': 'Passive Aggressive',
                        'xgb': 'XGBoost',
                        'rf': 'Random Forest',
                        'tree': 'Decision Tree'
                    }

                    embedding_name = embedding_mapping.get(self.best_supervised["feature_model"],
                                                          self.best_supervised["feature_model"])
                    classifier_name = classifier_mapping.get(self.best_supervised["classifier"],
                                                            self.best_supervised["classifier"])

                    fig.suptitle(f'Confusion Matrices - Best Supervised Model\n'
                                f'{embedding_name} + {classifier_name} (Macro F1: {self.best_supervised["macro_f1"]:.4f})',
                                fontsize=14, fontweight='bold', y=0.995)
                    fig.tight_layout(rect=[0, 0, 1, 0.99])
                    self._save_figure(fig, '06_confusion_matrices_supervised', dpi=300)
                    plt.close(fig)

                    print("  ✓ Charts saved: 06_confusion_matrices_supervised.png/.pdf")

                    # Generate histogram/prediction distribution for Supervised
                    print(f"\n  📊 Generating prediction distribution (Supervised)...")

                    n_rows_hist = int(np.ceil(n_labels / 4))
                    n_cols_hist = min(4, n_labels)

                    fig_hist, axes_hist = plt.subplots(n_rows_hist, n_cols_hist, figsize=(4 * n_cols_hist, 3 * n_rows_hist))
                    axes_hist = axes_hist.flatten() if n_labels > 1 else [axes_hist]

                    for idx, label_name in enumerate(label_names[:len(axes_hist)]):
                        ax = axes_hist[idx]

                        # Get predictions for this label
                        y_pred_label = y_pred[:, idx] if y_pred.ndim > 1 else (y_pred == idx).astype(int)

                        # Count predictions
                        pred_counts = np.bincount(y_pred_label.astype(int), minlength=2)

                        ax.bar(['Negative', 'Positive'], pred_counts, color=['#e74c3c', '#3498db'], alpha=0.7)
                        ax.set_title(label_name, fontsize=10, fontweight='bold')
                        ax.set_ylabel('Count', fontsize=9)
                        ax.grid(axis='y', alpha=0.3)

                        # Add counts on bars
                        for i, count in enumerate(pred_counts):
                            ax.text(i, count + max(pred_counts)*0.02, str(count),
                                   ha='center', fontsize=9, fontweight='bold')

                    # Hide extra subplots
                    for idx in range(n_labels, len(axes_hist)):
                        axes_hist[idx].axis('off')

                    fig_hist.suptitle(f'Prediction Distribution - Best Supervised Model\n'
                                f'{embedding_name} + {classifier_name} (Macro F1: {self.best_supervised["macro_f1"]:.4f})',
                                fontsize=14, fontweight='bold', y=0.995)
                    fig_hist.tight_layout(rect=[0, 0, 1, 0.99])
                    self._save_figure(fig_hist, '06_prediction_distribution_supervised', dpi=300)
                    plt.close(fig_hist)

                    print("  ✓ Charts saved: 06_prediction_distribution_supervised.png/.pdf")

                    # Summary statistics
                    print("\n  📊 Per-Category Performance (Best Supervised Model):\n")
                    print("  " + "-"*70)
                    print(f"  {'Category':<30s} {'Precision':<12s} {'Recall':<12s} {'F1':<12s}")
                    print("  " + "-"*70)

                    if self.comparison_summary and 'category_performance' in self.comparison_summary:
                        cat_perf = self.comparison_summary['category_performance']
                        for cat, metrics in sorted(cat_perf.items(), key=lambda x: x[1]['F1'], reverse=True):
                            print(f"  {cat:<30s} {metrics['Precision']:<12.4f} {metrics['Recall']:<12.4f} {metrics['F1']:<12.4f}")

                    print("  " + "-"*70)

        # ========== LLM MODEL ==========
        if self.best_llm is None:
            print("\n  ⚠ Best LLM model not identified")
        else:
            # Load predictions for best LLM model
            strategy_dir = Path(self.best_llm.get('strategy_dir', ''))

            if not strategy_dir.exists():
                print(f"\n  ⚠ LLM strategy directory not found: {strategy_dir}")
            else:
                # Find classifications file
                classifications_files = list(strategy_dir.glob('classifications_*.json'))

                if not classifications_files:
                    print(f"\n  ⚠ No classifications file found in {strategy_dir}")
                else:
                    classifications_file = classifications_files[0]
                    print(f"\n  📂 Loading predictions (LLM): {classifications_file.name}")

                    try:
                        with open(classifications_file, 'r') as f:
                            classifications = json.load(f)

                        # Load metrics to get categories
                        metrics_file = strategy_dir / 'metrics_latest.json'
                        with open(metrics_file, 'r') as f:
                            metrics_data = json.load(f)

                        label_names = metrics_data.get('categories', [])

                        if not label_names:
                            print("  ⚠ Categories not found in metrics file")
                            return
                        else:
                            # Extract predicted labels and match with ground truth
                            y_true_llm = []
                            y_pred_llm = []
                            image_paths = []

                            for item in classifications:
                                if item['status'] != 'success':
                                    continue

                                # Get predicted labels
                                classification = item['classification']['classificacao_categorias']
                                pred_labels = []
                                for cat in label_names:
                                    # Handle case where category might not exist in classification
                                    if cat in classification:
                                        valor = classification[cat]['valor']
                                        # Handle NaN values - treat as 0 (negative)
                                        if valor is None or (isinstance(valor, float) and np.isnan(valor)):
                                            pred_labels.append(0)
                                        else:
                                            pred_labels.append(1 if valor == 1 else 0)
                                    else:
                                        pred_labels.append(0)
                                y_pred_llm.append(pred_labels)
                                image_paths.append(item['image_path'])

                            # Match with ground truth from test.csv
                            if self.df_test is not None:
                                for img_path in image_paths:
                                    # Find matching row in test dataset
                                    match = self.df_test[self.df_test['LocalPath'] == img_path]
                                    if len(match) > 0:
                                        # Handle NaN values in ground truth
                                        true_labels = []
                                        for cat in label_names:
                                            val = match.iloc[0][cat]
                                            if pd.isna(val):
                                                true_labels.append(0)
                                            else:
                                                true_labels.append(int(val))
                                        y_true_llm.append(true_labels)
                                    else:
                                        # If not found, append zeros
                                        y_true_llm.append([0] * len(label_names))

                            # Generate confusion matrices and distributions
                            if y_pred_llm:
                                y_pred_llm = np.array(y_pred_llm)
                                y_true_llm = np.array(y_true_llm) if y_true_llm else None
                                n_labels = len(label_names)

                                print(f"  ✓ Loaded {len(y_pred_llm)} predictions with {n_labels} categories")

                                # Generate confusion matrices if ground truth available
                                if y_true_llm is not None and len(y_true_llm) == len(y_pred_llm):
                                    print("  ✓ Ground truth available - generating confusion matrices")

                                    n_rows = int(np.ceil(np.sqrt(n_labels)))
                                    n_cols = int(np.ceil(n_labels / n_rows))

                                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
                                    axes = axes.flatten() if n_labels > 1 else [axes]

                                    for idx, label_name in enumerate(label_names[:len(axes)]):
                                        ax = axes[idx]

                                        # Binary confusion matrix for this label
                                        # Replace NaN with 0 before converting to int
                                        y_true_label = np.nan_to_num(y_true_llm[:, idx], nan=0.0).astype(int)
                                        y_pred_label = np.nan_to_num(y_pred_llm[:, idx], nan=0.0).astype(int)

                                        cm = confusion_matrix(y_true_label, y_pred_label)

                                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                                   xticklabels=['Negative', 'Positive'],
                                                   yticklabels=['Negative', 'Positive'],
                                                   cbar=False)

                                        ax.set_title(label_name, fontsize=10, fontweight='bold')
                                        ax.set_xlabel('Predicted', fontsize=9)
                                        ax.set_ylabel('True', fontsize=9)

                                    # Hide extra subplots
                                    for idx in range(n_labels, len(axes)):
                                        axes[idx].axis('off')

                                    # Format LLM model and strategy names
                                    model_name = self.best_llm["model_name"]
                                    if 'llama' in model_name.lower():
                                        if 'scout' in model_name.lower():
                                            model_display = 'Llama Scout'
                                        elif 'maverick' in model_name.lower():
                                            model_display = 'Llama Maverick'
                                        else:
                                            model_display = 'Llama'
                                    elif 'mistral' in model_name.lower():
                                        model_display = 'Mistral Small'
                                    else:
                                        model_display = model_name.split('_')[-1].capitalize()

                                    strategy_mapping = {
                                        'zero_shot_basic': 'Basic',
                                        'zero_shot_with_descriptions': 'Basic With Description',
                                        'zero_shot_cot': 'CoT',
                                        'zero_shot_dual_role': 'Dual Role'
                                    }
                                    strategy_display = strategy_mapping.get(self.best_llm["strategy"],
                                                                           self.best_llm["strategy"])

                                    fig.suptitle(f'Confusion Matrices - Best LLM Model\n'
                                               f'{model_display} ({strategy_display}) '
                                               f'(Macro F1: {self.best_llm["macro_f1"]:.4f})',
                                               fontsize=14, fontweight='bold', y=0.995)
                                    fig.tight_layout(rect=[0, 0, 1, 0.99])
                                    self._save_figure(fig, '06_confusion_matrices_llm', dpi=300)
                                    plt.close(fig)

                                    print("  ✓ Charts saved: 06_confusion_matrices_llm.png/.pdf")
                                else:
                                    print("  ⚠ Ground truth not available - showing prediction distributions only")

                                # Always generate prediction distribution chart
                                n_rows = int(np.ceil(n_labels / 4))
                                n_cols = min(4, n_labels)

                                fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
                                axes = axes.flatten() if n_labels > 1 else [axes]

                                for idx, label_name in enumerate(label_names[:len(axes)]):
                                    ax = axes[idx]

                                    # Count predictions
                                    # Replace NaN with 0 before converting to int
                                    pred_label = np.nan_to_num(y_pred_llm[:, idx], nan=0.0).astype(int)
                                    pred_counts = np.bincount(pred_label, minlength=2)

                                    ax.bar(['Negative', 'Positive'], pred_counts, color=['#e74c3c', '#3498db'], alpha=0.7)
                                    ax.set_title(label_name, fontsize=10, fontweight='bold')
                                    ax.set_ylabel('Count', fontsize=9)
                                    ax.grid(axis='y', alpha=0.3)

                                    # Add counts on bars
                                    for i, count in enumerate(pred_counts):
                                        ax.text(i, count + max(pred_counts)*0.02, str(count),
                                               ha='center', fontsize=9, fontweight='bold')

                                # Hide extra subplots
                                for idx in range(n_labels, len(axes)):
                                    axes[idx].axis('off')

                                fig.suptitle(f'Prediction Distribution - Best LLM Model\n'
                                           f'{model_display} ({strategy_display}) '
                                           f'(Macro F1: {self.best_llm["macro_f1"]:.4f})',
                                           fontsize=14, fontweight='bold', y=0.995)
                                fig.tight_layout(rect=[0, 0, 1, 0.99])
                                self._save_figure(fig, '06_prediction_distribution_llm', dpi=300)
                                plt.close(fig)

                                print("  ✓ Charts saved: 06_prediction_distribution_llm.png/.pdf")

                                # Summary statistics from metrics
                                print("\n  📊 Per-Category Performance (Best LLM Model):\n")
                                print("  " + "-"*70)
                                print(f"  {'Category':<30s} {'Precision':<12s} {'Recall':<12s} {'F1':<12s}")
                                print("  " + "-"*70)

                                category_metrics = metrics_data.get('metrics', {})
                                sorted_categories = sorted(category_metrics.items(),
                                                         key=lambda x: x[1].get('f1', 0),
                                                         reverse=True)

                                for cat, metrics in sorted_categories:
                                    print(f"  {cat:<30s} {metrics.get('precision', 0):<12.4f} "
                                         f"{metrics.get('recall', 0):<12.4f} {metrics.get('f1', 0):<12.4f}")

                                print("  " + "-"*70)

                    except Exception as e:
                        print(f"  ⚠ Error loading LLM predictions: {e}")

    def analyze_false_positives_negatives(self):
        """6.5. Analysis: False positives and negatives for best models"""
        print("\n" + "="*80)
        print("6.5. ANALYSIS: FALSE POSITIVES AND NEGATIVES")
        print("="*80)

        # ========== SUPERVISED MODEL ==========
        if self.best_supervised is None:
            print("  ⚠ Best supervised model not identified")
            fp_sup_data = None
            fn_sup_data = None
        else:
            supervised_path = (
                self.results_path / 'supervised_classifiers' /
                self.best_supervised['feature_model'] /
                self.best_supervised['classifier'] /
                'test_predictions.npz'
            )

            if not supervised_path.exists():
                print(f"  ⚠ Predictions file not found for supervised model")
                fp_sup_data = None
                fn_sup_data = None
            else:
                data = np.load(supervised_path, allow_pickle=True)
                y_true_sup = data['y_true']
                y_pred_sup = data['y_pred']
                label_names = data['categories'].tolist() if 'categories' in data else None

                if label_names is None:
                    print("  ⚠ Categories not found for supervised model")
                    fp_sup_data = None
                    fn_sup_data = None
                else:
                    # Calculate false positives and negatives per category
                    fp_counts = []
                    fn_counts = []

                    for idx, cat in enumerate(label_names):
                        y_true_cat = y_true_sup[:, idx].astype(int)
                        y_pred_cat = y_pred_sup[:, idx].astype(int)

                        # False positives: predicted 1, true 0
                        fp = np.sum((y_pred_cat == 1) & (y_true_cat == 0))
                        # False negatives: predicted 0, true 1
                        fn = np.sum((y_pred_cat == 0) & (y_true_cat == 1))

                        fp_counts.append({'Category': cat, 'Count': fp})
                        fn_counts.append({'Category': cat, 'Count': fn})

                    fp_sup_data = pd.DataFrame(fp_counts).sort_values('Count', ascending=False)
                    fn_sup_data = pd.DataFrame(fn_counts).sort_values('Count', ascending=False)

                    print("\n  📊 Top 5 Categories with Most False Positives (Supervised):\n")
                    for idx, row in fp_sup_data.head(5).iterrows():
                        print(f"    {row['Category']:30s}: {row['Count']:.0f} false positives")

                    print("\n  📊 Top 5 Categories with Most False Negatives (Supervised):\n")
                    for idx, row in fn_sup_data.head(5).iterrows():
                        print(f"    {row['Category']:30s}: {row['Count']:.0f} false negatives")

        # ========== LLM MODEL ==========
        if self.best_llm is None:
            print("\n  ⚠ Best LLM model not identified")
            fp_llm_data = None
            fn_llm_data = None
        else:
            strategy_dir = Path(self.best_llm.get('strategy_dir', ''))

            if not strategy_dir.exists():
                print(f"\n  ⚠ LLM strategy directory not found")
                fp_llm_data = None
                fn_llm_data = None
            else:
                classifications_files = list(strategy_dir.glob('classifications_*.json'))

                if not classifications_files or self.df_test is None:
                    print(f"\n  ⚠ Cannot compute FP/FN for LLM (missing data)")
                    fp_llm_data = None
                    fn_llm_data = None
                else:
                    try:
                        with open(classifications_files[0], 'r') as f:
                            classifications = json.load(f)

                        metrics_file = strategy_dir / 'metrics_latest.json'
                        with open(metrics_file, 'r') as f:
                            metrics_data = json.load(f)

                        label_names = metrics_data.get('categories', [])

                        if not label_names:
                            print("  ⚠ Categories not found in metrics file")
                            fp_llm_data = None
                            fn_llm_data = None
                            return

                        # Extract predictions and ground truth
                        y_pred_llm_list = []
                        y_true_llm_list = []

                        for item in classifications:
                            if item['status'] != 'success':
                                continue

                            classification = item['classification']['classificacao_categorias']
                            pred_labels = []
                            for cat in label_names:
                                if cat in classification:
                                    valor = classification[cat]['valor']
                                    # Handle NaN values - treat as 0 (negative)
                                    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
                                        pred_labels.append(0)
                                    else:
                                        pred_labels.append(1 if valor == 1 else 0)
                                else:
                                    pred_labels.append(0)
                            y_pred_llm_list.append(pred_labels)

                            # Get ground truth
                            img_path = item['image_path']
                            match = self.df_test[self.df_test['LocalPath'] == img_path]
                            if len(match) > 0:
                                # Handle NaN values in ground truth
                                true_labels = []
                                for cat in label_names:
                                    val = match.iloc[0][cat]
                                    if pd.isna(val):
                                        true_labels.append(0)
                                    else:
                                        true_labels.append(int(val))
                                y_true_llm_list.append(true_labels)
                            else:
                                y_true_llm_list.append([0] * len(label_names))

                        y_pred_llm = np.array(y_pred_llm_list)
                        y_true_llm = np.array(y_true_llm_list)

                        # Calculate false positives and negatives per category
                        fp_counts = []
                        fn_counts = []

                        for idx, cat in enumerate(label_names):
                            y_true_cat = y_true_llm[:, idx].astype(int)
                            y_pred_cat = y_pred_llm[:, idx].astype(int)

                            fp = np.sum((y_pred_cat == 1) & (y_true_cat == 0))
                            fn = np.sum((y_pred_cat == 0) & (y_true_cat == 1))

                            fp_counts.append({'Category': cat, 'Count': fp})
                            fn_counts.append({'Category': cat, 'Count': fn})

                        fp_llm_data = pd.DataFrame(fp_counts).sort_values('Count', ascending=False)
                        fn_llm_data = pd.DataFrame(fn_counts).sort_values('Count', ascending=False)

                        print("\n  📊 Top 5 Categories with Most False Positives (LLM):\n")
                        for idx, row in fp_llm_data.head(5).iterrows():
                            print(f"    {row['Category']:30s}: {row['Count']:.0f} false positives")

                        print("\n  📊 Top 5 Categories with Most False Negatives (LLM):\n")
                        for idx, row in fn_llm_data.head(5).iterrows():
                            print(f"    {row['Category']:30s}: {row['Count']:.0f} false negatives")

                    except Exception as e:
                        print(f"\n  ⚠ Error computing FP/FN for LLM: {e}")
                        fp_llm_data = None
                        fn_llm_data = None

        # ========== VISUALIZATION ==========
        if fp_sup_data is not None or fp_llm_data is not None:
            fig, axes = plt.subplots(2, 2, figsize=(20, 14))
            axes = np.array(axes).reshape(2, 2)

            def plot_false_counts(ax, data, title, color):
                """Render horizontal bar chart or show placeholder when data is missing."""
                ax.set_title(title, fontsize=13, fontweight='bold')

                if data is None or data.empty:
                    ax.axis('off')
                    ax.text(
                        0.5,
                        0.5,
                        'No data available',
                        ha='center',
                        va='center',
                        fontsize=11,
                        transform=ax.transAxes,
                        fontweight='bold',
                        alpha=0.7,
                    )
                    return

                top_data = data.head(5)
                y_pos = np.arange(len(top_data))
                ax.barh(y_pos, top_data['Count'], color=color, alpha=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_data['Category'])
                ax.invert_yaxis()
                ax.set_xlabel('Count', fontsize=12, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)

                max_count = top_data['Count'].max() if not top_data.empty else 0
                offset = max(max_count * 0.02, 0.5) if max_count > 0 else 0.5
                for idx, value in enumerate(top_data['Count']):
                    ax.text(
                        value + offset,
                        idx,
                        f'{value:.0f}',
                        va='center',
                        fontsize=10,
                        fontweight='bold',
                    )

            # Top row: best supervised model
            plot_false_counts(
                axes[0, 0],
                fp_sup_data,
                'Top 5 False Positives\n(Best Supervised Model)',
                '#e74c3c',
            )
            plot_false_counts(
                axes[0, 1],
                fn_sup_data,
                'Top 5 False Negatives\n(Best Supervised Model)',
                '#f39c12',
            )

            # Bottom row: best LLM
            plot_false_counts(
                axes[1, 0],
                fp_llm_data,
                'Top 5 False Positives\n(Best LLM Model)',
                '#3498db',
            )
            plot_false_counts(
                axes[1, 1],
                fn_llm_data,
                'Top 5 False Negatives\n(Best LLM Model)',
                '#9b59b6',
            )

            fig.suptitle(
                'False Positives and Negatives Analysis - Best Models',
                fontsize=16,
                fontweight='bold',
            )
            fig.tight_layout()
            self._save_figure(fig, '06c_false_positives_negatives', dpi=300)
            plt.close(fig)

            print("\n  ✓ Charts saved: 06c_false_positives_negatives.png/.pdf")

    def analyze_qualitative_error_cases(self, seed: int = 43, n_samples: int = 6):
        """6.6. Random qualitative comparison between best supervised and LLM models."""
        print("\n" + "="*80)
        print("6.6. QUALITATIVE COMPARISON (RANDOM SAMPLES)")
        print("="*80)

        supervised_payload = self._load_best_supervised_predictions()
        if supervised_payload is None:
            print("  ⚠ Unable to load predictions for the best supervised model.")
            return

        llm_payload = self._load_best_llm_predictions(supervised_payload['categories'])
        if llm_payload is None:
            print("  ⚠ Unable to load predictions for the best LLM model.")
            return

        categories = supervised_payload['categories']
        paths = supervised_payload['paths']
        y_true = supervised_payload['y_true']
        y_pred_sup = supervised_payload['y_pred']

        ground_truth_map = {path: y_true[idx] for idx, path in enumerate(paths)}
        supervised_map = {path: y_pred_sup[idx] for idx, path in enumerate(paths)}

        llm_predictions_raw = llm_payload['predictions']
        llm_map = {
            path: preds for path, preds in llm_predictions_raw.items()
            if path in ground_truth_map
        }

        if not llm_map:
            print("  ⚠ No overlapping samples between LLM predictions and supervised ground truth.")
            return

        shared_paths = list(llm_map.keys())
        rng = random.Random(seed)
        sample_size = min(n_samples, len(shared_paths))
        if sample_size == 0:
            print("  ⚠ Not enough overlapping samples to display qualitative comparison.")
            return
        selected_paths = rng.sample(shared_paths, sample_size)

        def label_list(vector: np.ndarray) -> str:
            labels = [
                categories[idx].replace('\n', ' / ').strip()
                for idx, val in enumerate(vector)
                if int(val) == 1
            ]
            return ", ".join(labels) if labels else "Nenhuma"

        printable_rows = []
        samples_for_plot: List[Dict] = []

        for path in selected_paths:
            gt_vec = np.array(ground_truth_map[path], copy=True)
            sup_vec = np.array(supervised_map[path], copy=True)
            llm_vec = np.array(llm_map[path], copy=True)

            printable_rows.append({
                'image_path': path,
                'Ground Truth': label_list(gt_vec),
                'Supervisionado': label_list(sup_vec),
                'LLM': label_list(llm_vec),
            })

            samples_for_plot.append(
                {
                    'image_path': path,
                    'ground_truth_vector': gt_vec,
                    'primary_vector': sup_vec,
                    'secondary_vector': llm_vec,
                    'primary_label_name': "Supervisionado",
                    'secondary_label_name': "LLM",
                    'scenario_title': "Comparação Aleatória",
                }
            )

        df = pd.DataFrame(printable_rows)
        print(f"\n  ▶ Random comparison ({sample_size} samples, seed={seed})")
        print(df.to_string(index=False))

        self._render_qualitative_panels(
            samples=samples_for_plot,
            categories=categories,
            title="Random Comparison — Supervised vs LLM",
            filename_slug="random_comparison",
            add_row_dividers=True,
        )

    def _render_qualitative_panels(
        self,
        samples: List[Dict],
        categories: List[str],
        title: str,
        filename_slug: str,
        add_row_dividers: bool = False,
    ) -> None:
        if not samples:
            return

        def active_labels(vector: np.ndarray) -> str:
            labels = [
                categories[idx].replace('\n', ' / ').strip()
                for idx, val in enumerate(vector)
                if int(val) == 1
            ]
            return ", ".join(labels) if labels else "Nenhuma"

        n_samples = len(samples)
        cols = 2
        rows = math.ceil(n_samples / cols)
        fig_width = 8 * cols
        fig_height = 4.5 * rows
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        axes = np.array(axes).reshape(rows, cols)
        flat_axes = axes.flatten()

        for idx, sample in enumerate(samples):
            ax = flat_axes[idx]
            resolved_path = self._resolve_image_path(sample['image_path'])

            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

            if resolved_path and resolved_path.exists():
                try:
                    image = Image.open(resolved_path).convert("RGB")
                    ax.imshow(image)
                except Exception as exc:
                    ax.text(0.5, 0.5, f"Failed to open image:\n{exc}", ha='center', va='center')
            else:
                ax.text(0.5, 0.5, "Image not found", ha='center', va='center')

            gt_vector = sample['ground_truth_vector']
            primary_vector = sample['primary_vector']
            secondary_vector = sample['secondary_vector']

            category_display = sample.get('category_focus', '')
            primary_label = sample.get('primary_label_name', 'Modelo A')
            secondary_label = sample.get('secondary_label_name', 'Modelo B')
            scenario_title = sample.get('scenario_title', title)
            gt_focus = sample.get('ground_truth_value')
            primary_focus = sample.get('primary_value')
            secondary_focus = sample.get('secondary_value')

            text_lines: List[str] = []

            if category_display:
                text_lines.append(f"Target category: {category_display}")
                if gt_focus is not None:
                    text_lines.append(f"Ground Truth ({category_display}): {gt_focus}")
                if primary_focus is not None:
                    text_lines.append(f"{primary_label} ({category_display}): {primary_focus}")
                if secondary_focus is not None:
                    text_lines.append(f"{secondary_label} ({category_display}): {secondary_focus}")
                text_lines.append("")

            text_lines.extend(
                [
                    f"Ground Truth (all classes): {active_labels(gt_vector)}",
                    f"{primary_label} (all classes): {active_labels(primary_vector)}",
                    f"{secondary_label} (all classes): {active_labels(secondary_vector)}",
                ]
            )

            caption_ax = ax.inset_axes([0, -0.28, 1, 0.28])
            caption_ax.axis('off')
            caption_ax.text(
                0,
                0.9,
                "\n".join(text_lines) if text_lines else "No labels available",
                va='top',
                ha='left',
                fontsize=10,
                wrap=True,
            )

        for idx in range(n_samples, len(flat_axes)):
            flat_axes[idx].axis('off')

        fig.tight_layout(rect=[0, 0, 1, 0.95])

        for col in range(cols - 1):
            ax_left = axes[0, col]
            ax_right = axes[0, col + 1]
            pos_left = ax_left.get_position()
            pos_right = ax_right.get_position()
            x_line = (pos_left.x1 + pos_right.x0) / 2.0
            fig.add_artist(
                Line2D(
                    [x_line, x_line],
                    [0.08, 0.94],
                    transform=fig.transFigure,
                    color="black",
                    linewidth=2.5,
                    alpha=1.0,
                )
            )

        if add_row_dividers and rows > 1:
            for row in range(1, rows):
                ax_above = axes[row - 1, 0]
                ax_below = axes[row, 0]
                pos_above = ax_above.get_position()
                pos_below = ax_below.get_position()
                y_line = (pos_above.y0 + pos_below.y1) / 2.0
                fig.add_artist(
                    Line2D(
                        [0.03, 0.97],
                        [y_line, y_line],
                        transform=fig.transFigure,
                        color="black",
                        linewidth=2.5,
                        alpha=1.0,
                    )
                )

        slug = re.sub(r'[^a-z0-9]+', '_', filename_slug.lower()).strip('_')
        filename = f"06d_{slug}"
        self._save_figure(fig, filename, dpi=250)
        plt.close(fig)

        print(f"     ✓ Painel visual salvo: {filename}.png/.pdf")

    def analyze_additional_insights(self):
        """7. Analysis: Additional useful insights"""
        print("\n" + "="*80)
        print("7. ANALYSIS: ADDITIONAL INSIGHTS")
        print("="*80)

        # A. Correlation between metrics
        print("\n  🔗 A. Correlation Analysis\n")

        metrics_cols = ['Exact_Match', 'Jaccard', 'Micro_F1', 'Macro_F1', 'Weighted_F1', 'Hamming_Loss']

        # Create subplots for supervised and LLM
        n_plots = 0
        if self.df_supervised is not None:
            n_plots += 1
        if self.df_llm is not None:
            n_plots += 1

        if n_plots > 0:
            fig, axes = plt.subplots(1, n_plots, figsize=(10 * n_plots, 8))
            if n_plots == 1:
                axes = [axes]

            plot_idx = 0

            # Supervised correlation
            if self.df_supervised is not None:
                corr_matrix_sup = self.df_supervised[metrics_cols].corr()

                ax = axes[plot_idx]
                sns.heatmap(corr_matrix_sup, annot=True, fmt='.2f', cmap='coolwarm',
                           center=0, square=True, linewidths=1, ax=ax,
                           cbar_kws={'label': 'Correlation'})
                ax.set_title('Correlation Matrix\nSupervised Classifiers',
                            fontsize=14, fontweight='bold', pad=20)
                plot_idx += 1

            # LLM correlation
            if self.df_llm is not None:
                corr_matrix_llm = self.df_llm[metrics_cols].corr()

                ax = axes[plot_idx]
                sns.heatmap(corr_matrix_llm, annot=True, fmt='.2f', cmap='coolwarm',
                           center=0, square=True, linewidths=1, ax=ax,
                           cbar_kws={'label': 'Correlation'})
                ax.set_title('Correlation Matrix\nBest LLM Model',
                            fontsize=14, fontweight='bold', pad=20)

            fig.tight_layout()
            self._save_figure(fig, '07a_correlation_matrix', dpi=300)
            plt.close(fig)

            print("    ✓ Correlation matrix saved: 07a_correlation_matrix.png/.pdf")

        # B. Distribution of predictions quality
        print("\n  📈 B. Metrics Distribution Analysis\n")

        if self.df_combined is not None:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()

            metrics = ['Macro_F1', 'Micro_F1', 'Exact_Match', 'Jaccard']

            for idx, metric in enumerate(metrics):
                ax = axes[idx]

                for model_type in ['Supervised', 'LLM']:
                    data = self.df_combined[self.df_combined['Model_Type'] == model_type][metric].dropna()
                    ax.hist(data, alpha=0.6, label=model_type, bins=15)

                ax.set_xlabel(metric.replace('_', ' '), fontsize=11, fontweight='bold')
                ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
                ax.set_title(f'{metric.replace("_", " ")} Distribution', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)

            fig.suptitle('Metrics Distribution - Supervised vs LLM',
                        fontsize=16, fontweight='bold')
            fig.tight_layout()
            self._save_figure(fig, '07b_metrics_distribution', dpi=300)
            plt.close(fig)

            print("    ✓ Distribution analysis saved: 07b_metrics_distribution.png/.pdf")

        # C. Overall ranking
        print("\n  🏆 C. Overall Ranking (Top 10)\n")

        if self.df_combined is not None:
            ranking = self.df_combined.sort_values('Macro_F1', ascending=False).head(10)

            print("  " + "-"*90)
            print(f"  {'Rank':<6s} {'Model':<45s} {'Type':<12s} {'Macro F1':<15s}")
            print("  " + "-"*90)

            for idx, (_, row) in enumerate(ranking.iterrows(), 1):
                print(f"  {idx:<6d} {row['Model']:<45s} {row['Model_Type']:<12s} {row['Macro_F1']:<15.4f}")

            print("  " + "-"*90)

    def generate_summary_report(self):
        """Generate final summary report"""
        print("\n" + "="*80)
        print("GENERATING FINAL REPORT")
        print("="*80)

        summary = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'best_supervised': self.best_supervised,
            'best_llm': self.best_llm,
            'supervised_stats': {},
            'llm_stats': {},
            'comparison': {}
        }

        if self.df_supervised is not None:
            summary['supervised_stats'] = {
                'total_methods': len(self.df_supervised),
                'mean_macro_f1': float(self.df_supervised['Macro_F1'].mean()),
                'std_macro_f1': float(self.df_supervised['Macro_F1'].std()),
                'max_macro_f1': float(self.df_supervised['Macro_F1'].max()),
                'min_macro_f1': float(self.df_supervised['Macro_F1'].min()),
            }

        if self.df_llm is not None:
            summary['llm_stats'] = {
                'total_configurations': len(self.df_llm),
                'mean_macro_f1': float(self.df_llm['Macro_F1'].mean()),
                'std_macro_f1': float(self.df_llm['Macro_F1'].std()),
                'max_macro_f1': float(self.df_llm['Macro_F1'].max()),
                'min_macro_f1': float(self.df_llm['Macro_F1'].min()),
            }

        if self.df_combined is not None:
            summary['comparison'] = {
                'supervised_mean_macro_f1': float(
                    self.df_combined[self.df_combined['Model_Type'] == 'Supervised']['Macro_F1'].mean()
                ),
                'llm_mean_macro_f1': float(
                    self.df_combined[self.df_combined['Model_Type'] == 'LLM']['Macro_F1'].mean()
                ),
            }

        # Save report
        output_file = self.output_path / 'classification_analysis_report.json'
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"  ✓ Report saved: classification_analysis_report.json")

        # Print summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)

        if self.best_supervised:
            print(f"\n  🏆 Best Supervised Model:")
            print(f"      → Method: {self.best_supervised['method']}")
            print(f"      → Embedding: {self.best_supervised['feature_model']}")
            print(f"      → Classifier: {self.best_supervised['classifier']}")
            print(f"      → Macro F1: {self.best_supervised['macro_f1']:.4f}")

        if self.best_llm:
            print(f"\n  🏆 Best LLM Configuration:")
            print(f"      → Model: {self.best_llm['model_name']}")
            print(f"      → Strategy: {self.best_llm['strategy']}")
            print(f"      → Macro F1: {self.best_llm['macro_f1']:.4f}")

        print("\n" + "="*80)
        print(f"All results saved in: {self.output_path}")
        print("="*80)

    def run_all_analyses(self):
        """Execute all analyses"""
        print("\n" + "="*80)
        print("STARTING COMPLETE CLASSIFICATION ANALYSIS")
        print("="*80)

        self.analyze_embeddings_by_classifier()
        self.analyze_llm_comparison()
        self.analyze_supervised_vs_llm()
        self.analyze_category_difficulty()
        self.analyze_prompts_by_llm()
        self.analyze_confusion_matrices()
        self.analyze_false_positives_negatives()
        self.analyze_qualitative_error_cases()
        self.analyze_additional_insights()
        self.generate_summary_report()

        print("\n" + "="*80)
        print("✓ COMPLETE ANALYSIS FINISHED!")
        print("="*80)


def main():
    """Main function"""
    analyzer = ClassificationAnalyzer()
    analyzer.run_all_analyses()


if __name__ == '__main__':
    main()
