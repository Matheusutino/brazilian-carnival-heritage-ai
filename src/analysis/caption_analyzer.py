"""
Caption Analysis Script
Compare generated captions against ground-truth descriptions and compute text statistics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from numbers import Real
from typing import Dict, Iterable, List, Optional, Tuple
from pandas.api.types import CategoricalDtype

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
from matplotlib.lines import Line2D
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

# Optional dependency: sentence-transformers
try:
    from sentence_transformers import SentenceTransformer, util as st_util
except Exception:  # pragma: no cover - library may be unavailable
    SentenceTransformer = None
    st_util = None

# Optional dependency: BERTScore
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except Exception:  # pragma: no cover - library may be unavailable
    BERTSCORE_AVAILABLE = False


@dataclass
class CaptionRecord:
    """Container for a single caption prediction."""

    family: str  # e.g. classical or llm
    model: str   # backend/model name
    variant: str  # optional subfolder (e.g. strategy)
    source_file: str
    image_path: Optional[str]
    image_name: Optional[str]
    caption_text: str
    caption_long_text: str
    caption_source: str
    status: str


class CaptionAnalyzer:
    """Analyze captioning results."""

    MODEL_DISPLAY_NAMES = {
        "blip": "BLIP",
        "git": "GIT",
        "vit-gpt2": "ViT-GPT2",
        "meta_llama-4-maverick-17b-128e-instruct": "LLaMA 4 - Maverick",
        "meta_llama-4-scout-17b-16e-instruct": "LLaMA 4 - Scout",
        "mistralai_mistral-small-3.1-24b-instruct-2503": "Mistral Small 3.1",
    }
    MODEL_SORT_ORDER = [
        "blip",
        "git",
        "vit-gpt2",
        "meta_llama-4-maverick-17b-128e-instruct",
        "meta_llama-4-scout-17b-16e-instruct",
        "mistralai_mistral-small-3.1-24b-instruct-2503",
    ]

    def __init__(
        self,
        results_path: str = "results/captions",
        dataset_csv: str = "dataset/catalogacao_with_local_paths.csv",
        reference_column: str = "Descrição (informal)",
        output_path: Optional[str] = None,
    ) -> None:
        self.results_path = Path(results_path)
        self.dataset_csv = Path(dataset_csv)
        self.reference_column = reference_column
        self.output_path = Path(output_path) if output_path else self.results_path / "analysis"
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.dataset_df: Optional[pd.DataFrame] = None
        self.dataset_map: Dict[str, str] = {}
        self.records: List[CaptionRecord] = []
        self.embedding_model: object = None
        self.use_bertscore: bool = False

        self._load_dataset()
        self._load_results()

    # ------------------------------------------------------------------ #
    # Naming helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def _normalize_model_key(cls, name: Optional[str]) -> str:
        """
        Normalize a model name to a standardized key format.

        Args:
            name: Raw model name from file path or configuration

        Returns:
            Normalized model key without extensions
        """
        if not name:
            return ""
        key = str(name).replace("\\", "/").split("/")[-1]

        lower_key = key.lower()
        known_exts = [".json", ".csv", ".pt", ".bin", ".ckpt"]
        for ext in known_exts:
            if lower_key.endswith(ext):
                key = key[: -len(ext)]
                break

        return key

    @classmethod
    def _format_model_name(cls, raw_name: Optional[str]) -> str:
        """
        Format a model name for display purposes.

        Args:
            raw_name: Raw model name from file path or configuration

        Returns:
            Formatted, human-readable model name
        """
        key = cls._normalize_model_key(raw_name)
        if not key:
            return "Unknown"
        if key in cls.MODEL_DISPLAY_NAMES:
            return cls.MODEL_DISPLAY_NAMES[key]

        candidate = key.replace("_", " ").replace("-", " ")
        candidate = re.sub(r"\s+", " ", candidate).strip()
        return candidate.title() if candidate else key

    @classmethod
    def _model_order_index(cls, raw_name: Optional[str]) -> int:
        """
        Get the sort order index for a model name.

        Args:
            raw_name: Raw model name from file path or configuration

        Returns:
            Integer index for sorting (models not in sort order go last)
        """
        key = cls._normalize_model_key(raw_name)
        if key in cls.MODEL_SORT_ORDER:
            return cls.MODEL_SORT_ORDER.index(key)
        return len(cls.MODEL_SORT_ORDER)

    # ------------------------------------------------------------------ #
    # Loading utilities
    # ------------------------------------------------------------------ #
    def _load_dataset(self) -> None:
        """Load dataset metadata for ground-truth captions."""
        if not self.dataset_csv.exists():
            print(f"⚠ Dataset CSV not found: {self.dataset_csv}")
            return

        try:
            df = pd.read_csv(self.dataset_csv)
        except UnicodeDecodeError:
            df = pd.read_csv(self.dataset_csv, encoding="latin-1")

        # Normalize paths so they can be matched against predictions
        df["LocalPath"] = df["LocalPath"].fillna("").astype(str).str.strip()
        df[self.reference_column] = df[self.reference_column].fillna("").astype(str).str.strip()

        self.dataset_df = df
        self.dataset_map = dict(zip(df["LocalPath"], df[self.reference_column]))

        print(f"✓ Dataset loaded ({len(df)} items) from {self.dataset_csv}")

    def _load_results(self) -> None:
        """Load caption predictions from results directory."""
        if not self.results_path.exists():
            print(f"⚠ Results directory not found: {self.results_path}")
            return

        json_files = sorted(self.results_path.glob("**/captions_*.json"))
        if not json_files:
            print(f"⚠ No captions_*.json files found in {self.results_path}")
            return

        for json_path in json_files:
            relative_parts = json_path.relative_to(self.results_path).parts
            family = relative_parts[0] if len(relative_parts) >= 1 else "unknown"
            model = relative_parts[1] if len(relative_parts) >= 2 else "unknown"
            variant = relative_parts[2] if len(relative_parts) >= 3 else ""

            try:
                with open(json_path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
            except json.JSONDecodeError as exc:
                print(f"⚠ Failed to load {json_path}: {exc}")
                continue

            for entry in data:
                caption_text, caption_source, caption_long_text = self._extract_caption(entry)
                if not caption_text:
                    continue

                image_path = entry.get("image_path") or entry.get("metadata", {}).get("local_path")
                image_name = entry.get("image_name")
                status = entry.get("status", "unknown")

                self.records.append(
                    CaptionRecord(
                        family=family,
                        model=model,
                        variant=variant,
                        source_file=str(json_path.relative_to(self.results_path)),
                        image_path=image_path,
                        image_name=image_name,
                        caption_text=caption_text.strip(),
                        caption_long_text=caption_long_text.strip(),
                        caption_source=caption_source,
                        status=status,
                    )
                )

        print(f"✓ {len(self.records)} captions loaded from {len(json_files)} files")

    def _extract_caption(self, entry: dict) -> Tuple[str, str, str]:
        """Extract caption text from entry, handling heterogeneous formats."""
        preferred_fields = [
            "caption",
            "caption_pt",
            "descricao_curta",
            "descricao_longa",
            "generated_caption",
        ]

        caption = entry.get("caption")
        long_fields = ["descricao_longa", "descricao_completa", "descricao_detalhada"]
        caption_long_text = ""
        if isinstance(caption, str):
            return caption, "caption", ""

        if isinstance(caption, dict):
            for long_field in long_fields:
                value = caption.get(long_field)
                if isinstance(value, str) and value.strip():
                    caption_long_text = value
                    break

            for field in preferred_fields:
                value = caption.get(field)
                if isinstance(value, str) and value.strip():
                    return value, f"caption.{field}", caption_long_text

        # Top-level fallback fields
        for field in preferred_fields:
            value = entry.get(field)
            if isinstance(value, str) and value.strip():
                return value, field, ""

        return "", "", ""

    def _resolve_image_path(self, image_path: Optional[str]) -> Optional[Path]:
        """Return a readable Path for the given image, if it exists."""
        if not image_path:
            return None

        candidates = []
        path = Path(image_path)
        candidates.append(path)

        # When path is relative, ensure we also test against project root explicitly
        if not path.is_absolute():
            candidates.append(Path.cwd() / path)

            dataset_root = self.dataset_csv.parent
            if not str(path).startswith(str(dataset_root)):
                candidates.append(dataset_root / path)

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        print(f"⚠ Image not found: {image_path}")
        return None

    # ------------------------------------------------------------------ #
    # Metric computation
    # ------------------------------------------------------------------ #
    def _tokenize(self, text: str) -> List[str]:
        """Basic tokenizer suitable for BLEU/ROUGE without external resources."""
        return re.findall(r"\w+(?:'\w+)?", text.lower())

    def _compute_supervised_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute BLEU and ROUGE metrics against references."""
        if "reference_caption" not in df.columns:
            return df

        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        smoothing = SmoothingFunction().method3

        bleu_weights = {
            "bleu_1": (1, 0, 0, 0),
            "bleu_2": (0.5, 0.5, 0, 0),
            "bleu_3": (1 / 3, 1 / 3, 1 / 3, 0),
            "bleu_4": (0.25, 0.25, 0.25, 0.25),
        }

        metrics: Dict[str, List[float]] = {key: [] for key in ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_1", "rouge_l"]}

        for _, row in df.iterrows():
            text = row["caption_text"]
            reference = row["reference_caption"]

            if not reference:
                for key in metrics:
                    metrics[key].append(math.nan)
                continue

            hyp_tokens = self._tokenize(text)
            ref_tokens = self._tokenize(reference)

            if not hyp_tokens or not ref_tokens:
                for key in metrics:
                    metrics[key].append(0.0)
                continue

            for key, weights in bleu_weights.items():
                weights = tuple(weights)  # ensure tuple for sentence_bleu
                # sentence_bleu expects references as list of token lists
                try:
                    score = sentence_bleu(
                        [ref_tokens],
                        hyp_tokens,
                        weights=weights,
                        smoothing_function=smoothing,
                    )
                except ZeroDivisionError:
                    score = 0.0
                metrics[key].append(float(score))

            rouge = scorer.score(reference, text)
            metrics["rouge_1"].append(float(rouge["rouge1"].fmeasure))
            metrics["rouge_l"].append(float(rouge["rougeL"].fmeasure))

        for key, values in metrics.items():
            df[key] = values

        return df

    def _compute_embedding_similarity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute cosine similarity between prediction and reference embeddings."""
        if not self.embedding_model or "reference_caption" not in df.columns:
            return df

        similarities = []
        for _, row in df.iterrows():
            text = row["caption_text"]
            reference = row["reference_caption"]
            if not reference:
                similarities.append(math.nan)
                continue

            try:
                embedding_pred = self.embedding_model.encode(text, convert_to_tensor=True, show_progress_bar=False)
                embedding_ref = self.embedding_model.encode(reference, convert_to_tensor=True, show_progress_bar=False)
            except Exception as exc:  # pragma: no cover - device issues
                print(f"⚠ Failed to compute embeddings: {exc}")
                similarities.append(math.nan)
                continue

            similarity = float(st_util.cos_sim(embedding_pred, embedding_ref).cpu().item())
            similarities.append(similarity)

        df["embedding_similarity"] = similarities
        return df

    def _compute_bertscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute BERTScore for Portuguese captions."""
        if not self.use_bertscore or not BERTSCORE_AVAILABLE or "reference_caption" not in df.columns:
            return df

        # Filter rows with valid references
        valid_refs = df["reference_caption"].notna() & (df["reference_caption"] != "")
        if not valid_refs.any():
            return df

        candidates = df.loc[valid_refs, "caption_text"].tolist()
        references = df.loc[valid_refs, "reference_caption"].tolist()

        try:
            print("📊 Computing BERTScore (Portuguese)...")
            # Use multilingual BERT model optimized for Portuguese
            _, _, F1 = bert_score(
                candidates,
                references,
                lang="pt",
                model_type="bert-base-multilingual-cased",
                verbose=False,
                device="cuda" if self._has_cuda() else "cpu"
            )

            # Initialize column with NaN
            df["bertscore_f1"] = math.nan

            # Fill valid scores
            df.loc[valid_refs, "bertscore_f1"] = F1.cpu().numpy()

            print(f"✓ BERTScore computed for {len(candidates)} captions")
        except Exception as exc:
            print(f"⚠ Failed to compute BERTScore: {exc}")

        return df

    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _compute_unsupervised_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute metrics independent of ground truth."""
        stats = {
            "word_count": [],
            "unique_words": [],
            "unique_ratio": [],
            "char_count": [],
            "avg_word_length": [],
        }

        for _, row in df.iterrows():
            tokens = self._tokenize(row["caption_text"])
            word_count = len(tokens)
            unique_words = len(set(tokens))
            char_count = len(row["caption_text"])
            avg_word_length = (sum(len(token) for token in tokens) / word_count) if word_count else 0.0
            unique_ratio = (unique_words / word_count) if word_count else 0.0

            stats["word_count"].append(word_count)
            stats["unique_words"].append(unique_words)
            stats["unique_ratio"].append(unique_ratio)
            stats["char_count"].append(char_count)
            stats["avg_word_length"].append(avg_word_length)

        for key, values in stats.items():
            df[key] = values

        return df

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def enable_embeddings(self, model_name: str) -> None:
        """Load optional embedding model for semantic similarity."""
        if not SentenceTransformer:
            print("⚠ sentence-transformers not available; embedding metric disabled.")
            return

        try:
            self.embedding_model = SentenceTransformer(model_name)
            print(f"✓ Embedding model loaded: {model_name}")
        except Exception as exc:
            print(f"⚠ Failed to load embedding model '{model_name}': {exc}")
            self.embedding_model = None

    def enable_bertscore(self) -> None:
        """Enable BERTScore metric for Portuguese captions."""
        if not BERTSCORE_AVAILABLE:
            print("⚠ bert-score not available; install with: pip install bert-score")
            return

        self.use_bertscore = True
        print("✓ BERTScore enabled for Portuguese")

    def build_dataframe(self) -> pd.DataFrame:
        """Build dataframe combining predictions and references."""
        if not self.records:
            return pd.DataFrame()

        df = pd.DataFrame([record.__dict__ for record in self.records])

        if self.dataset_map:
            df["reference_caption"] = df["image_path"].map(self.dataset_map).fillna("")
        else:
            df["reference_caption"] = ""

        df = self._compute_unsupervised_metrics(df)

        if self.dataset_map:
            df = self._compute_supervised_metrics(df)
            df = self._compute_embedding_similarity(df)
            df = self._compute_bertscore(df)

        # Useful flags for aggregation
        df["has_reference"] = df["reference_caption"].astype(bool)

        return df

    def summarize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate metrics by model."""
        if df.empty:
            return df

        metric_columns = [
            col
            for col in [
                "bleu_1",
                "bleu_2",
                "bleu_3",
                "bleu_4",
                "rouge_1",
                "rouge_l",
                "embedding_similarity",
                "bertscore_f1",
                "word_count",
                "unique_ratio",
                "avg_word_length",
            ]
            if col in df.columns
        ]

        aggregation = {col: "mean" for col in metric_columns}
        aggregation["caption_text"] = "count"
        aggregation["has_reference"] = "sum"

        summary = (
            df.groupby(["family", "model", "variant"])
            .agg(aggregation)
            .reset_index()
            .rename(columns={"caption_text": "n_samples", "has_reference": "n_with_reference"})
        )

        return summary

    def save_outputs(self, df: pd.DataFrame, summary: pd.DataFrame) -> Tuple[Path, Path]:
        """Persist detailed and summary outputs."""
        detailed_path = self.output_path / "caption_metrics.csv"
        summary_path = self.output_path / "caption_summary.csv"

        df.to_csv(detailed_path, index=False)
        summary.to_csv(summary_path, index=False)

        print(f"✓ Detailed results saved to {detailed_path}")
        print(f"✓ Summary saved to {summary_path}")

        return detailed_path, summary_path

    # ------------------------------------------------------------------ #
    # Visualization methods
    # ------------------------------------------------------------------ #
    def plot_metric_distributions(self, df: pd.DataFrame, save: bool = True) -> Optional[Path]:
        """Create boxplot distributions of metrics by model family."""
        if df.empty:
            return None

        metric_columns = [
            col
            for col in ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_1", "rouge_l", "embedding_similarity", "bertscore_f1"]
            if col in df.columns
        ]

        if not metric_columns:
            print("⚠ No metrics available for visualization")
            return None

        # Calculate grid dimensions
        n_metrics = len(metric_columns)
        n_cols = 4
        n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]

        for idx, metric in enumerate(metric_columns):
            ax = axes[idx]
            data_to_plot = []
            tick_labels = []

            for family in df["family"].unique():
                family_data = df[df["family"] == family][metric].dropna()
                if len(family_data) > 0:
                    data_to_plot.append(family_data)

                    # Legend adjustment
                    label = family.upper() if family.lower() == "llm" else family.capitalize()
                    tick_labels.append(label)


            if data_to_plot:
                bp = ax.boxplot(data_to_plot, tick_labels=tick_labels, patch_artist=True)

                # Color boxes
                colors = ["#3498db", "#e74c3c", "#2ecc71"]
                for patch, color in zip(bp["boxes"], colors[: len(data_to_plot)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)

            ax.set_title(metric.upper().replace("_", "-"), fontweight="bold")
            ax.set_ylabel("Score")
            ax.grid(axis="y", alpha=0.3)

        # Hide unused subplots
        for idx in range(len(metric_columns), len(axes)):
            axes[idx].set_visible(False)

        #fig.suptitle("Metric Distributions by Model Family", fontsize=16, fontweight="bold", y=0.995)
        plt.tight_layout()

        if save:
            plot_path = self.output_path / "metric_distributions.pdf"
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            print(f"✓ Metric distribution plot saved: {plot_path}")
            return plot_path
        else:
            plt.show()
            return None

    def plot_model_comparison(self, summary: pd.DataFrame, save: bool = True) -> Optional[Path]:
        """Create horizontal bar chart comparing models by average metric score."""
        if summary.empty:
            return None

        metric_columns = [
            col
            for col in ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_1", "rouge_l", "embedding_similarity", "bertscore_f1"]
            if col in summary.columns
        ]

        if not metric_columns:
            print("⚠ No metrics available for comparison")
            return None

        # Calculate average score
        summary_copy = summary.copy()
        summary_copy["average_score"] = summary_copy[metric_columns].mean(axis=1)
        summary_copy["model_display"] = summary_copy["model"].apply(self._format_model_name)
        summary_copy = summary_copy.sort_values("average_score", ascending=True)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, max(8, len(summary_copy) * 0.5)))
        colors = summary_copy["family"].map({"classical": "#3498db", "llm": "#e74c3c"}).fillna("#95a5a6")
        bars = ax.barh(
            summary_copy["model_display"],
            summary_copy["average_score"],
            color=colors,
            alpha=0.7,
            label=None  # impede legenda automática
        )


        # Ajustar eixos e valores
        max_score = summary_copy["average_score"].max()
        ax.set_xlim(0, max_score * 1.1 if max_score > 0 else 1)
        for bar, score in zip(bars, summary_copy["average_score"]):
            ax.text(
                score + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}",
                va="center",
                fontsize=10,
                fontweight="bold",
            )


        # Legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#3498db", alpha=0.7, label="Classical"),
            Patch(facecolor="#e74c3c", alpha=0.7, label="LLM"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()

        if save:
            plot_path = self.output_path / "model_comparison.pdf"
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            print(f"✓ Model comparison plot saved: {plot_path}")
            return plot_path
        else:
            plt.show()
            return None

    def plot_metrics_per_model(self, summary: pd.DataFrame, save: bool = True) -> Optional[Path]:
        """Plot metric profiles for each model to compare their strengths."""
        if summary.empty:
            return None

        metric_columns = [
            col
            for col in ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_1", "rouge_l",
                        "embedding_similarity", "bertscore_f1"]
            if col in summary.columns
        ]

        if not metric_columns:
            print("⚠ No metrics available for per-model visualization")
            return None

        profile_df = summary.copy()
        profile_df["model_key"] = profile_df["model"].apply(self._normalize_model_key)
        profile_df["model_display"] = profile_df["model"].apply(self._format_model_name)

        # -------- METRICS IN CAPS --------
        rename_dict = {m: m.upper() for m in metric_columns}
        profile_df.rename(columns=rename_dict, inplace=True)
        metric_columns_caps = list(rename_dict.values())

        # -------- LONG DF --------
        long_df = (
            profile_df[["model_display", "family"] + metric_columns_caps]
            .melt(id_vars=["model_display", "family"], value_vars=metric_columns_caps,
                    var_name="metric", value_name="score")
            .dropna(subset=["score"])
        )

        if long_df.empty:
            print("⚠ No per-model metric data available")
            return None

        # -------- Marker selection by family --------
        marker_map = {"classical": "o", "llm": "X"}

        # -------- Custom order --------
        custom_model_order = [
            "BLIP", "ViT-GPT2", "GIT",
            "Mistral Small 3.1", "LLaMA 4 - Scout", "LLaMA 4 - Maverick"
        ]

        all_models_in_df = long_df["model_display"].unique()

        ordered_categories = [
            model for model in custom_model_order
            if model in all_models_in_df
        ]
        ordered_categories.extend(
            [model for model in all_models_in_df if model not in ordered_categories]
        )

        model_cat_type = pd.CategoricalDtype(categories=ordered_categories, ordered=True)
        long_df["model_display"] = long_df["model_display"].astype(model_cat_type)

        # -------- RENAME FOR LEGEND --------
        plot_df = long_df.rename(columns={"model_display": "Model", "family": "Family"})

        # -------- PLOT --------
        fig, ax = plt.subplots(
            figsize=(max(12, len(metric_columns_caps)*1.5),
                        max(8, len(profile_df)*0.6))
        )

        sns.lineplot(
            data=plot_df,
            x="metric",
            y="score",
            hue="Model",
            style="Family",
            markers=marker_map,
            dashes=False,
            linewidth=2.5,
            markersize=9,
            ax=ax
        )

        # -------- LABELS --------
        ax.set_xlabel("Metrics", fontsize=13, fontweight="bold", labelpad=15)
        ax.set_ylabel("Score", fontsize=13, fontweight="bold")

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", alpha=0.3)

        # -------- LEGEND CONFIGURATION (IN-PLOT UPPER LEFT) --------
        ax.legend(
            loc="upper left",          # Position inside the plot
            frameon=True,              # Enable box
            fancybox=True,             
            shadow=False,
            ncol=1,                    
            fontsize=10,
            edgecolor="gray",          # Box border color
            borderpad=1
        )

        fig.tight_layout()

        if save:
            plot_path = self.output_path / "model_metric_profiles.pdf"
            fig.savefig(plot_path, bbox_inches="tight")
            plt.close()
            print(f"✓ Model metric profiles plot saved: {plot_path}")
            return plot_path
        else:
            plt.show()
            return None



    def plot_family_metrics_comparison(self, df: pd.DataFrame, save: bool = True) -> Optional[Path]:
        """Create grouped bar chart comparing Classical vs LLM across all metrics."""
        if df.empty:
            return None

        metric_columns = [
            col
            for col in [
                "bleu_1",
                "bleu_2",
                "bleu_3",
                "bleu_4",
                "rouge_1",
                "rouge_l",
                "embedding_similarity",
                "bertscore_f1",
            ]
            if col in df.columns
        ]

        if not metric_columns:
            print("⚠ No metrics available for family comparison")
            return None

        # Calculate mean metrics by family
        family_means = df.groupby("family")[metric_columns].mean().T

        if family_means.empty:
            print("⚠ No family data available")
            return None

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(metric_columns))
        width = 0.35

        families = family_means.columns.tolist()
        colors = {"classical": "#3498db", "llm": "#e74c3c"}

        bars = []
        for idx, family in enumerate(families):
            offset = width * (idx - len(families) / 2 + 0.5)
            bar = ax.bar(
                x + offset,
                family_means[family],
                width,
                label=("LLM" if family.lower() == "llm" else family.capitalize()),
                color=colors.get(family, "#95a5a6"),
                alpha=0.8,
            )
            bars.append(bar)

            # Add value labels on bars
            for i, v in enumerate(family_means[family]):
                if not np.isnan(v):
                    ax.text(
                        i + offset,
                        v + 0.02,
                        f"{v:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=15,
                        fontweight="bold",
                    )

        ax.set_xlabel("Metrics", fontsize=13, fontweight="bold")
        ax.set_ylabel("Average Score", fontsize=13, fontweight="bold")
        #ax.set_title("Classical vs LLM: Average Metrics Comparison", fontsize=15, fontweight="bold", pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper().replace("_", "-") for m in metric_columns], rotation=45, ha="right")
        ax.legend(loc="upper left", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(family_means.max()) * 1.15)

        plt.tight_layout()

        if save:
            plot_path = self.output_path / "family_metrics_comparison.pdf"
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            print(f"✓ Family metrics comparison plot saved: {plot_path}")
            return plot_path
        else:
            plt.show()
            return None

    def plot_random_caption_examples(
        self,
        df: pd.DataFrame,
        blip_model: str = "blip",
        llama_model: str = "meta_llama-4-maverick-17b-128e-instruct",
        n_samples: int = 3,
        seed: Optional[int] = 42,
        save: bool = True,
    ) -> Optional[Path]:
        """Display random images with BLIP and LLaMA captions side-by-side."""
        if df.empty:
            return None

        target_models = {blip_model, llama_model}
        available = df[df["model"].isin(target_models) & df["image_path"].astype(bool)]

        if available.empty:
            print("⚠ No records found for the requested models.")
            return None

        comparisons = []
        for image_path, group in available.groupby("image_path"):
            captions = {}
            for _, row in group.iterrows():
                captions[row["model"]] = {
                    "short": row["caption_text"],
                    "long": row.get("caption_long_text", ""),
                }

            if not target_models.issubset(captions.keys()):
                continue

            resolved_path = self._resolve_image_path(image_path)
            if not resolved_path:
                continue

            image_name = group["image_name"].iloc[0] if "image_name" in group.columns else Path(image_path).name
            reference_caption = group["reference_caption"].iloc[0] if "reference_caption" in group.columns else ""
            comparisons.append(
                {
                    "image_file": resolved_path,
                    "image_label": image_name or Path(image_path).name,
                    "captions": captions,
                    "reference_caption": reference_caption,
                }
            )

        if not comparisons:
            print("⚠ No images found with captions from both models.")
            return None

        rng = random.Random(seed)
        sample_count = min(n_samples, len(comparisons))
        if sample_count <= 0:
            print("⚠ Not enough examples to display.")
            return None

        if len(comparisons) > sample_count:
            selected = rng.sample(comparisons, sample_count)
        else:
            selected = comparisons

        fig, axes = plt.subplots(sample_count, 2, figsize=(14, 4 * sample_count))
        if sample_count == 1:
            axes = np.array([axes])

        def _wrap(text: str, width: int = 70) -> str:
            clean = text.strip() if text else ""
            return textwrap.fill(clean, width=width) if clean else "(no caption)"

        def _display_name(name: str) -> str:
            return self._format_model_name(name)

        for idx, (ax_img, ax_text) in enumerate(axes):
            example = selected[idx]
            try:
                image = Image.open(example["image_file"]).convert("RGB")
            except Exception as exc:
                ax_img.axis("off")
                ax_img.text(0.5, 0.5, f"Failed to open image:\n{exc}", ha="center", va="center")
            else:
                ax_img.imshow(image)
                #ax_img.set_title(example["image_label"], fontsize=12, fontweight="bold")
                ax_img.axis("off")

            ax_text.axis("off")
            captions = example["captions"]
            blip_info = captions.get(blip_model, {})
            llama_info = captions.get(llama_model, {})

            blip_short_raw = blip_info.get("short", "")
            llama_short_raw = llama_info.get("short", "")
            llama_long_raw = llama_info.get("long", "")
            reference_raw = example.get("reference_caption", "")

            blip_text = _wrap(blip_short_raw).capitalize()
            llama_text = _wrap(llama_short_raw)
            llama_long = _wrap(llama_long_raw) if llama_long_raw else ""
            reference_text = _wrap(reference_raw)

           
            sections = []

            if reference_raw and reference_raw.strip():
                sections.append(f"Original Caption:\n{reference_text}")

            sections.append(f"{_display_name(blip_model)}:\n{blip_text}")
            sections.append(f"{_display_name(llama_model)} - Short Description:\n{llama_text}")

            if llama_long_raw and llama_long_raw.strip() and llama_long != llama_text:
                sections.append(f"{_display_name(llama_model)} - Full Description:\n{llama_long}")

            text_block = "\n\n".join(sections)
            ax_text.text(0, 1, text_block, va="top", ha="left", fontsize=11, transform=ax_text.transAxes)

        plt.tight_layout()

        # Now, draw the lines in the exact space *between* the subplots.
        if sample_count > 1:
            for idx in range(1, sample_count):
                # Get the text axis *above* the line
                ax_above = axes[idx - 1, 1]
                pos_above = ax_above.get_position()  # Bbox([x0, y0, width, height])

                # Get the text axis *below* the line
                ax_below = axes[idx, 1]
                pos_below = ax_below.get_position()

                # The line 'y' is the midpoint between the *bottom* (y0) of the upper axis
                # and the *top* (y1) of the lower axis.
                # (pos_below.y1 is pos_below.y0 + pos_below.height)
                y_line = (pos_above.y0 + pos_below.y1) / 2.0

                line = Line2D(
                    [0.02, 0.98],
                    [y_line, y_line],         # Use the calculated y_line
                    transform=fig.transFigure,
                    color="#000000",          # Change to black
                    linewidth=2,
                    alpha=1.0,                # Change to opaque
                )
                fig.add_artist(line)

        if save:
            plot_path = self.output_path / "random_caption_examples.pdf"
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            print(f"✓ Random caption examples saved: {plot_path}")
            return plot_path
        else:
            plt.show()
            return None


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyzes caption results.")
    parser.add_argument(
        "--results",
        type=str,
        default="results/captions",
        help="Directory containing captions_*.json files",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/catalogacao_with_local_paths.csv",
        help="CSV with original descriptions (column 'LocalPath').",
    )
    parser.add_argument(
        "--reference-column",
        type=str,
        default="Descrição (informal)",
        help="CSV column used as reference.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save results (default: results/captions/analysis).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-transformers model name for semantic similarity.",
    )
    parser.add_argument(
        "--no-bertscore",
        action="store_true",
        help="Disable BERTScore (enabled by default).",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Disable embeddings (enabled by default).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    analyzer = CaptionAnalyzer(
        results_path=args.results,
        dataset_csv=args.dataset,
        reference_column=args.reference_column,
        output_path=args.output,
    )

    # Enable embeddings by default
    if not args.no_embeddings:
        analyzer.enable_embeddings(args.embedding_model)

    # Enable BERTScore by default
    if not args.no_bertscore:
        analyzer.enable_bertscore()

    df = analyzer.build_dataframe()
    if df.empty:
        print("⚠ No data available for analysis.")
        return

    summary = analyzer.summarize(df)
    analyzer.save_outputs(df, summary)

    # Generate visualizations
    print("\n=== GENERATING VISUALIZATIONS ===")
    analyzer.plot_metric_distributions(df)
    analyzer.plot_model_comparison(summary)
    analyzer.plot_metrics_per_model(summary)
    analyzer.plot_family_metrics_comparison(df)
    analyzer.plot_random_caption_examples(df)

    if not summary.empty:
        print("\n=== MODEL SUMMARY ===")
        print(
            summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}" if isinstance(x, Real) else str(x),
            )
        )


if __name__ == "__main__":
    main()
