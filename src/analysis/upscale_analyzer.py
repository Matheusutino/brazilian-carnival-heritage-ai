"""
Upscale Analysis Script
Complete analysis of upscaling results: Classical vs Deep Learning
"""

import json
import warnings
from math import pi, ceil
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

warnings.filterwarnings('ignore')

# Visualization configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class UpscaleAnalyzer:
    """Upscaling results analyzer"""

    def __init__(
        self,
        results_path: str = 'results/upscale',
        dataset_path: str = 'dataset'
    ):
        self.results_path = Path(results_path)
        self.dataset_path = Path(dataset_path)
        self.output_path = self.results_path / 'analysis'
        self.output_path.mkdir(exist_ok=True, parents=True)

        # Loaded data
        self.all_metrics: Dict = {}
        self.df_metrics: Optional[pd.DataFrame] = None
        self.df_normalized: Optional[pd.DataFrame] = None
        self.df_ranking: Optional[pd.DataFrame] = None
        self.df_tradeoff: Optional[pd.DataFrame] = None

        # Metrics
        self.metric_cols = ['Sharpness', 'Edge_Strength', 'Contrast', 'Oversharp', 'Ringing', 'Self_Consistency']
        self.quality_metrics = ['Sharpness', 'Edge_Strength', 'Contrast', 'Self_Consistency']
        self.artifact_metrics = ['Oversharp', 'Ringing']
        self.composite_metrics = ['Sharpness', 'Edge_Strength', 'Contrast', 'Oversharp', 'Ringing']
        self.zoom_region = (0.35, 0.45, 0.55, 0.65)  # (left, top, right, bottom) in relative coordinates

        self._load_data()

    def _load_data(self):
        """
        Load all upscale metrics data from result files.

        Scans the results directory for *_metrics.json files, extracts method information,
        and organizes metrics by method type (classical/deep_learning) and scale factor.
        """
        print("Loading upscale data...")

        # Load metrics from all methods
        if not self.results_path.exists():
            print(f"  ⚠ Directory not found: {self.results_path}")
            return

        for metrics_file in self.results_path.glob('*_metrics.json'):
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                method_name = data['method']

                # Extract method type and scale factor
                if '_2x' in method_name:
                    base_method = method_name.replace('_2x', '')
                    scale = '2x'
                elif '_4x' in method_name:
                    base_method = method_name.replace('_4x', '')
                    scale = '4x'
                else:
                    base_method = method_name
                    scale = 'unknown'

                # Determine method type
                classical_methods = ['nearest', 'bilinear', 'bicubic', 'lanczos']
                if base_method in classical_methods:
                    method_type = 'classical'
                else:
                    method_type = 'deep_learning'

                self.all_metrics[method_name] = {
                    'base_method': base_method,
                    'scale': scale,
                    'type': method_type,
                    'n_images': data['n_images'],
                    'n_successful': data['n_successful'],
                    'metrics': data['aggregated_metrics'],
                    'timestamp': data['timestamp']
                }

        print(f"  ✓ Total methods: {len(self.all_metrics)}")
        print(f"    - Classical: {sum(1 for m in self.all_metrics.values() if m['type'] == 'classical')}")
        print(f"    - Deep Learning: {sum(1 for m in self.all_metrics.values() if m['type'] == 'deep_learning')}")

        # Create DataFrame
        self._create_dataframes()

    def _create_dataframes(self):
        """
        Create DataFrames with all organized metrics.

        Transforms loaded metrics into pandas DataFrames for easier analysis,
        including method names, types, scales, and all metric values.
        """
        rows = []

        for method_name, data in self.all_metrics.items():
            metrics = data['metrics']
            row = {
                'Method': method_name,
                'Base_Method': data['base_method'],
                'Scale': data['scale'],
                'Type': data['type'],
                'Sharpness': metrics.get('sharpness', {}).get('mean', 0),
                'Edge_Strength': metrics.get('edge_strength', {}).get('mean', 0),
                'Contrast': metrics.get('contrast', {}).get('mean', 0),
                'Oversharp': metrics.get('oversharp', {}).get('mean', 0),
                'Ringing': metrics.get('ringing', {}).get('mean', 0),
                'Self_Consistency': metrics.get('self_consistency', {}).get('mean', 0),
                'Images': data['n_successful']
            }
            rows.append(row)

        self.df_metrics = pd.DataFrame(rows)
        self.df_metrics = self.df_metrics.sort_values(['Type', 'Scale', 'Sharpness'], ascending=[True, True, False])

        # Create normalized DataFrame
        self._normalize_metrics()

    def _normalize_metrics(self):
        """
        Normalize metrics to 0-1 scale for fair comparison.

        Quality metrics (higher is better) are normalized normally.
        Artifact metrics (lower is better) are inverted during normalization.
        """
        self.df_normalized = self.df_metrics.copy()

        for col in self.metric_cols:
            col_min = self.df_normalized[col].min()
            col_max = self.df_normalized[col].max()
            col_range = col_max - col_min

            if col_range == 0:
                self.df_normalized[col] = 0
            elif col in ['Oversharp', 'Ringing']:  # Invert for artifacts (lower is better)
                self.df_normalized[col] = 1 - (self.df_normalized[col] - col_min) / col_range
            else:  # For quality (higher is better)
                self.df_normalized[col] = (self.df_normalized[col] - col_min) / col_range

    def _save_figure(self, fig, filename_base: str, dpi: int = 300):
        """
        Save figure in both PNG and PDF formats.

        Args:
            fig: Matplotlib figure object to save
            filename_base: Base filename without extension
            dpi: Resolution for output images (default: 300)
        """
        png_path = self.output_path / f'{filename_base}.png'
        pdf_path = self.output_path / f'{filename_base}.pdf'
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
        fig.savefig(pdf_path, dpi=dpi, bbox_inches='tight')

    def analyze_descriptive_statistics(self):
        """
        Perform descriptive statistical analysis of upscaling metrics.

        Computes and displays statistics grouped by method type (classical/deep_learning)
        and scale factor (2x/4x), including mean, std, min, and max values.
        Also identifies the best performing method for each metric.
        """
        print("\n" + "="*80)
        print("1. ANALYSIS: DESCRIPTIVE STATISTICS")
        print("="*80)

        # Statistics by type
        print("\n📈 Statistics by Method Type\n")
        for method_type in ['classical', 'deep_learning']:
            subset = self.df_metrics[self.df_metrics['Type'] == method_type][self.metric_cols]
            print(f"\n{'='*70}")
            print(f"  {method_type.upper().replace('_', ' ')}")
            print(f"{'='*70}")
            print(subset.describe().T[['mean', 'std', 'min', 'max']])

        # Statistics by scale
        print("\n\n📈 Statistics by Scale Factor\n")
        for scale in ['2x', '4x']:
            subset = self.df_metrics[self.df_metrics['Scale'] == scale][self.metric_cols]
            if len(subset) > 0:
                print(f"\n{'='*70}")
                print(f"  SCALE {scale.upper()}")
                print(f"{'='*70}")
                print(subset.describe().T[['mean', 'std', 'min', 'max']])

        # Best method in each metric
        print("\n\n🏆 Best Method per Metric\n")
        for metric in self.quality_metrics:
            best_idx = self.df_metrics[metric].idxmax()
            best_method = self.df_metrics.loc[best_idx, 'Method']
            best_value = self.df_metrics.loc[best_idx, metric]
            method_type = self.df_metrics.loc[best_idx, 'Type']
            scale = self.df_metrics.loc[best_idx, 'Scale']
            print(f"  {metric:20s}: {best_method:30s} ({method_type}, {scale}) - {best_value:.2f}")

        print("\n  (For artifacts, lower is better)\n")
        for metric in self.artifact_metrics:
            best_idx = self.df_metrics[metric].idxmin()
            best_method = self.df_metrics.loc[best_idx, 'Method']
            best_value = self.df_metrics.loc[best_idx, metric]
            method_type = self.df_metrics.loc[best_idx, 'Type']
            scale = self.df_metrics.loc[best_idx, 'Scale']
            print(f"  {metric:20s}: {best_method:30s} ({method_type}, {scale}) - {best_value:.2f}")

    def analyze_all_metrics_comparison(self):
        """
        Create comprehensive comparison visualization of all metrics across methods.

        Generates grouped bar chart showing normalized scores for all metrics,
        allowing direct comparison of method performance across all dimensions.
        """
        print("\n" + "="*80)
        print("2. ANALYSIS: ALL METRICS COMPARISON")
        print("="*80)

        # --- control parameters ---
        bar_width = 0.2       # controls bar width
        group_spacing = 1.1    # controls spacing between groups (1 = default; <1 brings closer, >1 spreads apart)

        fig, ax = plt.subplots(figsize=(20, 8))

        # x-axis adjusted according to spacing between groups
        x = np.arange(len(self.df_normalized)) * group_spacing

        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']

        for i, metric in enumerate(self.metric_cols):
            offset = (i - len(self.metric_cols) / 2) * bar_width + bar_width / 2
            bars = ax.bar(
                x + offset,
                self.df_normalized[metric],
                bar_width,
                label=metric,
                color=colors[i % len(colors)],
                alpha=0.8
            )
            for bar, value in zip(bars, self.df_normalized[metric]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f'{value:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    fontweight='bold'
                )

        ax.set_xlabel('Method', fontsize=13, fontweight='bold')
        ax.set_ylabel('Normalized Score (0-1)', fontsize=13, fontweight='bold')
        ax.set_title('Comparison of All Metrics by Method (Normalized)', fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(self.df_normalized['Method'], rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper left', ncol=3, fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])

        fig.tight_layout()
        self._save_figure(fig, '01_all_metrics_comparison', dpi=300)
        plt.close(fig)

        print("  ✓ Charts saved: 01_all_metrics_comparison.png/.pdf")


    def analyze_scale_comparison(self):
        """
        Compare performance between 2x and 4x scale factors.

        Creates side-by-side bar charts for each metric, showing how methods
        perform at different upscaling ratios. Only includes methods available
        in both 2x and 4x configurations.
        """
        print("\n" + "="*80)
        print("3. ANALYSIS: 2X VS 4X COMPARISON")
        print("="*80)

        # Base methods that exist in both scales
        methods_2x = set(self.df_metrics[self.df_metrics['Scale'] == '2x']['Base_Method'].unique())
        methods_4x = set(self.df_metrics[self.df_metrics['Scale'] == '4x']['Base_Method'].unique())
        common_methods = sorted(list(methods_2x.intersection(methods_4x)))

        if len(common_methods) == 0:
            print("  ⚠ No common methods between 2x and 4x for comparison")
            return

        print(f"  ✓ Comparing {len(common_methods)} methods: {', '.join(common_methods)}")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        legend_handles = None

        # >>> main adjustments
        group_gap = 1.55      # spreads groups apart on x-axis
        width = 0.70          # thickens bars
        label_pad_pts = 4     # distance from number to top of bar (in points)

        for idx, metric in enumerate(self.metric_cols):
            ax = axes[idx]

            values_2x, values_4x = [], []

            for method in common_methods:
                row_2x = self.df_metrics[(self.df_metrics['Scale'] == '2x') & (self.df_metrics['Base_Method'] == method)]
                values_2x.append(row_2x[metric].iloc[0] if len(row_2x) > 0 else np.nan)

                row_4x = self.df_metrics[(self.df_metrics['Scale'] == '4x') & (self.df_metrics['Base_Method'] == method)]
                values_4x.append(row_4x[metric].iloc[0] if len(row_4x) > 0 else np.nan)

            # x with larger spacing between groups
            x = np.arange(len(common_methods), dtype=float) * group_gap

            # thicker bars
            bars_2x = ax.bar(x - width/2, values_2x, width, label='2x', color='#3498db', alpha=0.8)
            bars_4x = ax.bar(x + width/2, values_4x, width, label='4x', color='#e74c3c', alpha=0.8)

            if legend_handles is None and len(bars_2x) and len(bars_4x):
                legend_handles = (bars_2x[0], bars_4x[0])

            # labels above bars (without overlapping)
            def add_value_labels(bars, fmt="{:.2f}"):
                for b in bars:
                    h = b.get_height()
                    if np.isnan(h):
                        continue
                    ax.annotate(
                        fmt.format(h),
                        xy=(b.get_x() + b.get_width()/2, h),
                        xytext=(0, label_pad_pts),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=9, rotation=0, clip_on=False
                    )

            add_value_labels(bars_2x)
            add_value_labels(bars_4x)

            # gives space at the top for labels to fit
            finite_vals = [v for v in (values_2x + values_4x) if not np.isnan(v)]
            if finite_vals:
                ymax = max(finite_vals)
                ax.set_ylim(top=ymax * 1.15)

            ax.set_title(metric, fontweight='bold', fontsize=13)
            ax.set_xticks(x)
            ax.set_xticklabels(common_methods, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)

        fig.suptitle('Metrics Comparison: 2x vs 4x', fontsize=16, fontweight='bold', y=1.00)
        if legend_handles:
            fig.legend(
                legend_handles,
                ['2x', '4x'],
                loc='lower center',
                bbox_to_anchor=(0.5, -0.02),
                ncol=2,
                frameon=True
            )
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        self._save_figure(fig, '02_scale_comparison', dpi=300)
        plt.close(fig)

        print("  ✓ Charts saved: 02_scale_comparison.png/.pdf")


    def analyze_heatmap(self):
        """
        Generate heatmap visualization of all metrics across methods.

        Creates a color-coded matrix showing normalized metric scores for all methods,
        making it easy to identify patterns and compare performance at a glance.
        """
        print("\n" + "="*80)
        print("4. ANALYSIS: METRICS HEATMAP")
        print("="*80)

        fig, ax = plt.subplots(figsize=(14, 10))

        heatmap_data = self.df_normalized.set_index('Method')[self.metric_cols]

        sns.heatmap(heatmap_data.T, annot=True, fmt='.2f', cmap='RdYlGn',
                   cbar_kws={'label': 'Normalized Score'}, linewidths=0.5, ax=ax,
                   vmin=0, vmax=1)

        ax.set_title('Metrics Heatmap by Method (Normalized)', fontsize=15, fontweight='bold', pad=20)
        ax.set_xlabel('Method', fontsize=13, fontweight='bold')
        ax.set_ylabel('Metric', fontsize=13, fontweight='bold')

        fig.tight_layout()
        self._save_figure(fig, '03_metrics_heatmap', dpi=300)
        plt.close(fig)

        print("  ✓ Charts saved: 03_metrics_heatmap.png/.pdf")

    def analyze_radar_chart(self):
        """
        Create radar/spider charts for multidimensional metric comparison.

        Generates polar plots showing the performance profile of each method across
        all metrics, facilitating visual identification of strengths and weaknesses.
        Separate charts are created for 2x and 4x scale factors.
        """
        print("\n" + "="*80)
        print("5. ANALYSIS: RADAR CHART")
        print("="*80)

        scales = [scale for scale in ['2x', '4x'] if not self.df_normalized[self.df_normalized['Scale'] == scale].empty]

        if not scales:
            print("  ⚠ No methods available for radar chart")
            return

        categories = self.metric_cols
        xticklabels = [label.replace('_', '\n') for label in categories]
        angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
        angles_cycle = angles + angles[:1]

        for scale in scales:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})

            scale_df = self.df_normalized[self.df_normalized['Scale'] == scale].copy()

            if scale_df.empty:
                plt.close(fig)
                continue

            # Sort: classical methods first, then deep learning, both alphabetically
            scale_df['sort_key'] = scale_df['Type'].apply(lambda x: 0 if x == 'classical' else 1)
            scale_df = scale_df.sort_values(['sort_key', 'Method']).reset_index(drop=True)

            colors = plt.cm.Set2(np.linspace(0, 1, len(scale_df)))

            handles = []
            labels = []

            for color, (_, row) in zip(colors, scale_df.iterrows()):
                values = row[self.metric_cols].tolist()
                values += values[:1]
                line, = ax.plot(angles_cycle, values, linewidth=2, label=row['Method'], color=color)
                ax.fill(angles_cycle, values, alpha=0.15, color=color)
                handles.append(line)
                labels.append(row['Method'])

            ax.set_xticks(angles)
            ax.set_xticklabels(xticklabels, fontsize=11)
            ax.tick_params(axis='x', pad=15)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)

            # For 4x, use more columns to fit all in horizontal layout
            ncol = len(labels) if scale == '4x' else min(6, len(labels))

            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                       ncol=ncol, fontsize=9, frameon=True)

            fig.tight_layout(rect=(0, 0.05, 1, 1))
            self._save_figure(fig, f'04_radar_chart_{scale}', dpi=300)
            plt.close(fig)

            print(f"  ✓ Charts saved: 04_radar_chart_{scale}.png/.pdf")

    def analyze_classical_vs_dl(self):
        """
        Compare classical interpolation methods against deep learning approaches.

        Creates box plots showing the distribution of metric scores for classical
        vs deep learning methods, highlighting performance differences between
        traditional and modern upscaling techniques.
        """
        print("\n" + "="*80)
        print("6. ANALYSIS: CLASSICAL VS DEEP LEARNING")
        print("="*80)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, metric in enumerate(self.metric_cols):
            ax = axes[idx]

            data_to_plot = [
                self.df_metrics[self.df_metrics['Type'] == 'classical'][metric].values,
                self.df_metrics[self.df_metrics['Type'] == 'deep_learning'][metric].values
            ]

            bp = ax.boxplot(data_to_plot, labels=['Classical', 'Deep Learning'], patch_artist=True)

            colors = ['#3498db', '#e74c3c']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            ax.set_title(metric, fontweight='bold', fontsize=13)
            ax.set_ylabel('Score')
            ax.grid(axis='y', alpha=0.3)

        fig.suptitle('Metrics Distribution: Classical vs Deep Learning',
                     fontsize=16, fontweight='bold', y=1.00)
        fig.tight_layout()
        self._save_figure(fig, '05_classical_vs_dl', dpi=300)
        plt.close(fig)

        print("  ✓ Charts saved: 05_classical_vs_dl.png/.pdf")

    def analyze_quality_vs_artifacts(self):
        """
        Analyze the trade-off between quality improvements and artifact introduction.

        Creates scatter plots showing the relationship between quality metrics
        (sharpness, edge strength) and artifact metrics (oversharp, ringing),
        helping identify methods that achieve good balance.
        """
        print("\n" + "="*80)
        print("7. ANALYSIS: QUALITY VS ARTIFACTS")
        print("="*80)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # Sharpness vs Oversharp
        ax1 = axes[0]
        for method_type in ['classical', 'deep_learning']:
            subset = self.df_metrics[self.df_metrics['Type'] == method_type]
            color = '#3498db' if method_type == 'classical' else '#e74c3c'
            ax1.scatter(subset['Sharpness'], subset['Oversharp'],
                       label=method_type.replace('_', ' ').title(),
                       alpha=0.7, s=100, color=color)

            for _, row in subset.iterrows():
                ax1.annotate(row['Method'], (row['Sharpness'], row['Oversharp']),
                           fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')

        ax1.set_xlabel('Sharpness (higher = better)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Oversharp Artifacts (lower = better)', fontsize=12, fontweight='bold')
        ax1.set_title('Trade-off: Sharpness vs Over-sharpening', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Edge Strength vs Ringing
        ax2 = axes[1]
        for method_type in ['classical', 'deep_learning']:
            subset = self.df_metrics[self.df_metrics['Type'] == method_type]
            color = '#3498db' if method_type == 'classical' else '#e74c3c'
            ax2.scatter(subset['Edge_Strength'], subset['Ringing'],
                       label=method_type.replace('_', ' ').title(),
                       alpha=0.7, s=100, color=color)

            for _, row in subset.iterrows():
                ax2.annotate(row['Method'], (row['Edge_Strength'], row['Ringing']),
                           fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')

        ax2.set_xlabel('Edge Strength (higher = better)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Ringing Artifacts (lower = better)', fontsize=12, fontweight='bold')
        ax2.set_title('Trade-off: Edge Strength vs Ringing', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        self._save_figure(fig, '06_quality_vs_artifacts', dpi=300)
        plt.close(fig)

        print("  ✓ Charts saved: 06_quality_vs_artifacts.png/.pdf")

    def analyze_correlation(self):
        """
        Compute and visualize correlations between different metrics.

        Generates correlation matrix heatmap showing relationships between metrics,
        and identifies the strongest correlations to understand metric dependencies.
        """
        print("\n" + "="*80)
        print("8. ANALYSIS: METRICS CORRELATION")
        print("="*80)

        corr_matrix = self.df_metrics[self.metric_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, ax=ax,
                   cbar_kws={'label': 'Correlation'})

        ax.set_title('Correlation Matrix between Upscale Metrics',
                    fontsize=15, fontweight='bold', pad=20)

        fig.tight_layout()
        self._save_figure(fig, '07_correlation_matrix', dpi=300)
        plt.close(fig)

        # Show strongest correlations
        print("\n  🔗 Strongest Correlations:\n")
        corr_pairs = []
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                corr_pairs.append((self.metric_cols[i], self.metric_cols[j], corr_matrix.iloc[i, j]))

        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for metric1, metric2, corr_value in corr_pairs[:5]:
            print(f"    {metric1} <-> {metric2}: {corr_value:.3f}")

        print("  ✓ Charts saved: 07_correlation_matrix.png/.pdf")

    def analyze_ranking(self):
        """
        Create overall ranking of methods based on composite score.

        Calculates a composite score from normalized metrics (quality metrics
        positively weighted, artifacts negatively weighted) and ranks all methods.
        Generates bar chart visualization of the top performers.
        """
        print("\n" + "="*80)
        print("9. ANALYSIS: OVERALL RANKING")
        print("="*80)

        # Calculate composite score (without Self_Consistency)
        self.df_ranking = self.df_normalized.copy()
        self.df_ranking['Composite_Score'] = self.df_ranking[self.composite_metrics].mean(axis=1)
        self.df_ranking = self.df_ranking.sort_values('Composite_Score', ascending=False).reset_index(drop=True)
        self.df_ranking.index = self.df_ranking.index + 1
        self.df_ranking.index.name = 'Rank'

        print("\n  🏅 Top 10 Methods (by Composite Score):")
        print("  Metrics: Sharpness, Edge Strength, Contrast, Oversharp (inv), Ringing (inv)\n")
        for idx, row in self.df_ranking.head(10).iterrows():
            print(f"    {idx}. {row['Method']:30s} ({row['Type']:15s}, {row['Scale']}) - {row['Composite_Score']:.3f}")

        # Visualize ranking
        fig, ax = plt.subplots(figsize=(14, 10))

        colors = ['#2ecc71' if t == 'deep_learning' else '#3498db' for t in self.df_ranking['Type']]
        bars = ax.barh(self.df_ranking['Method'], self.df_ranking['Composite_Score'], color=colors, alpha=0.7)

        ax.set_xlabel('Composite Score (Normalized)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Method', fontsize=13, fontweight='bold')
        ax.set_title('Overall Ranking of Upscale Methods', fontsize=15, fontweight='bold', pad=20)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        for bar, score in zip(bars, self.df_ranking['Composite_Score']):
            ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', va='center', fontsize=9, fontweight='bold')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', alpha=0.7, label='Classical'),
            Patch(facecolor='#2ecc71', alpha=0.7, label='Deep Learning')
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        fig.tight_layout()
        self._save_figure(fig, '08_ranking', dpi=300)
        plt.close(fig)

        # Save CSV
        csv_path = self.output_path / '08_ranking.csv'
        self.df_ranking.to_csv(csv_path)

        print(f"  ✓ Charts saved: 08_ranking.png/.pdf")
        print(f"  ✓ CSV saved: 08_ranking.csv")

    def analyze_tradeoff(self):
        """
        Analyze trade-off between quality and artifact control across methods.

        Calculates quality and artifact scores, plots methods in 2D space to visualize
        the trade-off, and identifies methods with the best balance between high quality
        and low artifacts.
        """
        print("\n" + "="*80)
        print("10. ANALYSIS: TRADE-OFF ANALYSIS")
        print("="*80)

        self.df_tradeoff = self.df_normalized.copy()

        quality_metrics_list = ['Sharpness', 'Edge_Strength', 'Contrast']
        artifact_metrics_list = ['Oversharp', 'Ringing']

        self.df_tradeoff['Quality_Score'] = self.df_tradeoff[quality_metrics_list].mean(axis=1)
        self.df_tradeoff['Artifact_Score'] = self.df_tradeoff[artifact_metrics_list].mean(axis=1)
        self.df_tradeoff['Balance_Score'] = (self.df_tradeoff['Quality_Score'] + self.df_tradeoff['Artifact_Score']) / 2

        # Scatter plot
        fig, ax = plt.subplots(figsize=(14, 10))

        for method_type in ['classical', 'deep_learning']:
            subset = self.df_tradeoff[self.df_tradeoff['Type'] == method_type]
            color = '#3498db' if method_type == 'classical' else '#e74c3c'

            for scale in ['2x', '4x']:
                scale_subset = subset[subset['Scale'] == scale]
                marker = 'o' if scale == '2x' else 's'
                label = f"{method_type.replace('_', ' ').title()} - {scale}"

                ax.scatter(scale_subset['Quality_Score'], scale_subset['Artifact_Score'],
                          label=label, alpha=0.7, s=150, color=color, marker=marker, edgecolors='black')

                for _, row in scale_subset.iterrows():
                    ax.annotate(row['Method'], (row['Quality_Score'], row['Artifact_Score']),
                               fontsize=9, alpha=0.8, xytext=(5, 5), textcoords='offset points')

        ax.set_xlabel('Quality Score (higher = better)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Artifact Control (higher = better)', fontsize=13, fontweight='bold')
        ax.set_title('Trade-off Analysis: Quality vs Artifacts', fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        ax.text(0.75, 0.75, 'Ideal Zone', fontsize=12, ha='center', color='green', fontweight='bold', alpha=0.5)
        ax.text(0.25, 0.25, 'Worst Zone', fontsize=12, ha='center', color='red', fontweight='bold', alpha=0.5)

        fig.tight_layout()
        self._save_figure(fig, '09_tradeoff_analysis', dpi=300)
        plt.close(fig)

        # Top 5 by balance
        print("\n  ⚖️  Top 5 Methods with Best Balance:\n")
        for idx, row in self.df_tradeoff.nlargest(5, 'Balance_Score').iterrows():
            print(f"    {row['Method']:30s} - Balance: {row['Balance_Score']:.3f} " +
                  f"(Quality: {row['Quality_Score']:.3f}, Artifacts: {row['Artifact_Score']:.3f})")

        # Save CSV
        csv_path = self.output_path / '09_tradeoff_analysis.csv'
        self.df_tradeoff[['Method', 'Type', 'Scale', 'Quality_Score', 'Artifact_Score', 'Balance_Score']].to_csv(csv_path)

        print(f"  ✓ Charts saved: 09_tradeoff_analysis.png/.pdf")
        print(f"  ✓ CSV saved: 09_tradeoff_analysis.csv")

    def analyze_visual_examples(self):
        """
        Create side-by-side visual comparison of upscaling methods.

        Loads upscaled images and displays them in a grid layout with cropped/zoomed
        regions to highlight differences in quality and artifacts. Separate figures
        are generated for 2x and 4x upscaling.
        """
        print("\n" + "="*80)
        print("11. ANALYSIS: VISUAL EXAMPLES")
        print("="*80)

        upscaled_dir = self.dataset_path / 'upscaled'

        if not upscaled_dir.exists():
            print("  ⚠ Upscaled directory not found")
            return

        target_image_path = self.dataset_path / 'downloaded' / '1990 1.JPG'

        if not target_image_path.exists():
            print("  ⚠ Target original image not found: 1990 1.JPG")
            return

        base_stem = target_image_path.stem
        print(f"  → Selected image: {base_stem}")

        with Image.open(target_image_path) as img:
            original_image = img.convert('RGB')

        def crop_zoom(image: Image.Image) -> Image.Image:
            width, height = image.size
            left = int(self.zoom_region[0] * width)
            top = int(self.zoom_region[1] * height)
            right = int(self.zoom_region[2] * width)
            bottom = int(self.zoom_region[3] * height)
            left = max(0, min(left, width - 1))
            top = max(0, min(top, height - 1))
            right = max(left + 1, min(right, width))
            bottom = max(top + 1, min(bottom, height))
            return image.crop((left, top, right, bottom))

        # Create comparison figures for 2x and 4x
        for scale_factor in ['2x', '4x']:
            methods = [
                m for m, data in self.all_metrics.items() if data['scale'] == scale_factor
            ]
            methods.sort(
                key=lambda method: (
                    0 if self.all_metrics[method]['type'] == 'classical' else 1,
                    method
                )
            )

            if not methods:
                print(f"  ⚠ No methods found for {scale_factor}")
                continue

            # Load images
            comparisons = [{
                'label': 'Original',
                'image': original_image.copy(),
                'method': None,
                'type': 'reference'
            }]

            for method in methods:
                method_dir = upscaled_dir / method
                img_path = None
                for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                    potential_name = f"{base_stem}_{method}{ext}"
                    potential_path = method_dir / potential_name
                    if potential_path.exists():
                        img_path = potential_path
                        break

                if img_path and img_path.exists():
                    try:
                        with Image.open(img_path) as method_img:
                            comparisons.append({
                                'label': method,
                                'image': method_img.convert('RGB'),
                                'method': method,
                                'type': self.all_metrics[method]['type']
                            })
                    except Exception as e:  # pylint: disable=broad-except
                        print(f"    ⚠ Error loading {method}: {e}")
                else:
                    print(f"    ⚠ Missing upscale image for {method}: expected {base_stem}_{method}.[jpg|png]")

            if len(comparisons) == 1:
                print(f"  ⚠ No upscaled images loaded for {scale_factor}")
                continue

            n_items = len(comparisons)
            cols = 4 if n_items > 4 else max(2, n_items)
            rows = ceil(n_items / cols)

            fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
            if n_items == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if rows > 1 else axes

            for idx, item in enumerate(comparisons):
                cropped_img = crop_zoom(item['image'])
                ax = axes[idx]
                ax.imshow(cropped_img)
                if item['method'] is None:
                    title = "Original\n(Reference)"
                else:
                    title = f"{item['label']}\n({item['type']})"
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.axis('off')

            for idx in range(n_items, len(axes)):
                axes[idx].axis('off')

            fig.suptitle(
                f'Visual Comparison - {scale_factor} Upscaling\nImage: {base_stem}',
                fontsize=16,
                fontweight='bold'
            )
            fig.tight_layout()
            self._save_figure(fig, f'10_visual_comparison_{scale_factor}', dpi=200)
            plt.close(fig)

            print(f"  ✓ Charts saved: 10_visual_comparison_{scale_factor}.png/.pdf ({n_items} panels)")

    def generate_summary_report(self):
        """
        Generate comprehensive summary report of all analyses.

        Creates a JSON file containing key findings including best methods overall,
        best balance, best per metric, rankings, and analysis metadata.
        """
        print("\n" + "="*80)
        print("GENERATING FINAL REPORT")
        print("="*80)

        summary = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_methods': len(self.df_metrics),
            'classical_methods': len(self.df_metrics[self.df_metrics['Type'] == 'classical']),
            'deep_learning_methods': len(self.df_metrics[self.df_metrics['Type'] == 'deep_learning']),
            'best_overall': {
                'method': self.df_ranking.iloc[0]['Method'],
                'type': self.df_ranking.iloc[0]['Type'],
                'scale': self.df_ranking.iloc[0]['Scale'],
                'composite_score': float(self.df_ranking.iloc[0]['Composite_Score'])
            },
            'best_balance': {
                'method': self.df_tradeoff.nlargest(1, 'Balance_Score').iloc[0]['Method'],
                'type': self.df_tradeoff.nlargest(1, 'Balance_Score').iloc[0]['Type'],
                'scale': self.df_tradeoff.nlargest(1, 'Balance_Score').iloc[0]['Scale'],
                'balance_score': float(self.df_tradeoff.nlargest(1, 'Balance_Score').iloc[0]['Balance_Score'])
            },
            'best_per_metric': {},
            'rankings': self.df_ranking[['Method', 'Type', 'Scale', 'Composite_Score']].to_dict('records')
        }

        # Add best method per metric
        for metric in self.quality_metrics:
            best_idx = self.df_metrics[metric].idxmax()
            summary['best_per_metric'][metric] = {
                'method': self.df_metrics.loc[best_idx, 'Method'],
                'type': self.df_metrics.loc[best_idx, 'Type'],
                'scale': self.df_metrics.loc[best_idx, 'Scale'],
                'score': float(self.df_metrics.loc[best_idx, metric])
            }

        for metric in self.artifact_metrics:
            best_idx = self.df_metrics[metric].idxmin()
            summary['best_per_metric'][metric] = {
                'method': self.df_metrics.loc[best_idx, 'Method'],
                'type': self.df_metrics.loc[best_idx, 'Type'],
                'scale': self.df_metrics.loc[best_idx, 'Scale'],
                'score': float(self.df_metrics.loc[best_idx, metric])
            }

        # Save report
        output_file = self.output_path / 'summary_report.json'
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"  ✓ Report saved: summary_report.json")

        # Print summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        print(f"\nBest Overall Method:")
        print(f"  → {summary['best_overall']['method']}")
        print(f"  → Composite Score: {summary['best_overall']['composite_score']:.3f}")

        print(f"\nBest Balance (Quality + Artifact Control):")
        print(f"  → {summary['best_balance']['method']}")
        print(f"  → Balance Score: {summary['best_balance']['balance_score']:.3f}")

        print("\n" + "="*80)
        print(f"All results saved in: {self.output_path}")
        print("="*80)

    def run_all_analyses(self):
        """
        Execute complete suite of upscaling analyses.

        Runs all 11 analysis methods in sequence, generating visualizations,
        statistics, and reports for comprehensive comparison of upscaling methods.
        """
        print("\n" + "="*80)
        print("STARTING COMPLETE UPSCALING ANALYSIS")
        print("="*80)

        self.analyze_descriptive_statistics()
        self.analyze_all_metrics_comparison()
        self.analyze_scale_comparison()
        self.analyze_heatmap()
        self.analyze_radar_chart()
        self.analyze_classical_vs_dl()
        self.analyze_quality_vs_artifacts()
        self.analyze_correlation()
        self.analyze_ranking()
        self.analyze_tradeoff()
        self.analyze_visual_examples()
        self.generate_summary_report()

        print("\n" + "="*80)
        print("✓ COMPLETE ANALYSIS FINISHED!")
        print("="*80)


def main():
    """
    Main entry point for upscaling analysis script.

    Initializes the UpscaleAnalyzer with default paths and runs all analyses.
    """
    analyzer = UpscaleAnalyzer()
    analyzer.run_all_analyses()


if __name__ == '__main__':
    main()
