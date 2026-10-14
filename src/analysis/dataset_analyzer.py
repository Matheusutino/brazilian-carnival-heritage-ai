"""Exploratory analysis for the Suvaco de Cristo archive dataset.

Run from the project root with::

    python src/analysis/dataset_analyzer.py

By default, the script reads ``dataset/dataset_completo.csv`` and writes the
analysis artifacts to ``results/dataset/analysis``.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


DEFAULT_DATASET = Path("dataset/dataset_completo.csv")
DEFAULT_OUTPUT = Path("results/dataset/analysis")


def normalize_name(value: str) -> str:
    """Normalize a column name to simplify semantic matching."""
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower()


def find_column(columns: pd.Index, *terms: str) -> str | None:
    """Return the first column containing all requested normalized terms."""
    normalized_terms = [normalize_name(term) for term in terms]
    for column in columns:
        normalized_column = normalize_name(str(column))
        if all(term in normalized_column for term in normalized_terms):
            return str(column)
    return None


def json_value(value: Any) -> Any:
    """Convert pandas and NumPy values into JSON-compatible Python values."""
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def parse_catalog_number(series: pd.Series, thousands_separator: bool = False) -> pd.Series:
    """Parse numeric metadata, removing invisible Unicode formatting marks."""
    cleaned = series.astype("string").map(
        lambda value: "".join(
            character for character in str(value) if unicodedata.category(character) != "Cf"
        ).strip()
        if pd.notna(value)
        else pd.NA
    )
    if thousands_separator:
        cleaned = cleaned.str.replace(r"(?<=\d)\.(?=\d{3}$)", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


class DatasetAnalyzer:
    """Generate a reproducible exploratory analysis of an archive dataset."""

    def __init__(self, dataset_path: Path, output_path: Path) -> None:
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")

        self.dataset_path = dataset_path
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.df = pd.read_csv(dataset_path, low_memory=False)
        self.keyword_columns = self._detect_keyword_columns()
        self.valid_image_indices: list[int] = []

    def _detect_keyword_columns(self) -> list[str]:
        """Detect the binary keyword columns appended to the complete dataset."""
        local_path_column = find_column(self.df.columns, "localpath")
        candidates = list(self.df.columns)
        if local_path_column:
            start = candidates.index(local_path_column) + 1
            candidates = candidates[start:]

        keyword_columns: list[str] = []
        for column in candidates:
            numeric = pd.to_numeric(self.df[column], errors="coerce").dropna()
            if not numeric.empty and set(numeric.unique()).issubset({0, 1}):
                keyword_columns.append(str(column))
        return keyword_columns

    def _save_figure(self, figure: plt.Figure, name: str) -> None:
        """Save a titleless chart in raster and publication-ready PDF formats."""
        figure.suptitle("")
        for axis in figure.axes:
            axis.set_title("")
        figure.tight_layout()
        figure.savefig(self.output_path / f"{name}.png", dpi=180, bbox_inches="tight")
        figure.savefig(self.output_path / f"{name}.pdf", bbox_inches="tight")
        plt.close(figure)

    def _resolve_local_path(self, value: str) -> Path:
        """Resolve a catalog path from either the CWD or the dataset project root."""
        path = Path(value)
        if path.is_absolute() or path.exists():
            return path
        return self.dataset_path.resolve().parent.parent / path

    def analyze_quality(self) -> dict[str, Any]:
        """Analyze completeness, uniqueness, and local-file availability."""
        missing = self.df.isna().sum()
        missing_table = pd.DataFrame(
            {
                "coluna": self.df.columns,
                "tipo": self.df.dtypes.astype(str).values,
                "ausentes": missing.values,
                "percentual_ausente": (missing.values / len(self.df) * 100).round(2),
                "valores_unicos": self.df.nunique(dropna=True).values,
            }
        ).sort_values(["percentual_ausente", "coluna"], ascending=[False, True])
        missing_table.to_csv(self.output_path / "missing_values.csv", index=False)

        id_column = find_column(self.df.columns, "id")
        duplicate_ids = int(self.df[id_column].duplicated().sum()) if id_column else None

        local_path_column = find_column(self.df.columns, "localpath")
        local_files_found = None
        local_files_missing = None
        empty_local_paths = None
        if local_path_column:
            paths = self.df[local_path_column].dropna().astype(str)
            exists = paths.map(lambda value: Path(value).is_file())
            local_files_found = int(exists.sum())
            local_files_missing = int((~exists).sum())
            empty_local_paths = int(self.df[local_path_column].isna().sum())

        plot_data = (
            missing_table.query("percentual_ausente > 0")
            .head(20)
            .sort_values("percentual_ausente")
        )
        if not plot_data.empty:
            figure, axis = plt.subplots(figsize=(11, max(5, len(plot_data) * 0.35)))
            axis.barh(plot_data["coluna"], plot_data["percentual_ausente"], color="#c44e52")
            axis.set(
                title="Colunas com dados ausentes",
                xlabel="Missing values (%)",
                ylabel="Dataset columns",
            )
            self._save_figure(figure, "01_missing_values")

        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "duplicate_ids": duplicate_ids,
            "total_missing_cells": int(self.df.isna().sum().sum()),
            "missing_cells_percent": round(float(self.df.isna().mean().mean() * 100), 2),
            "local_files_found": local_files_found,
            "local_files_missing": local_files_missing,
            "empty_local_paths": empty_local_paths,
        }

    def analyze_descriptive_statistics(self) -> None:
        """Save numeric statistics and a compact profile for every column."""
        numeric = self.df.select_dtypes(include="number")
        if not numeric.empty:
            numeric.describe().T.to_csv(
                self.output_path / "numeric_summary.csv", index_label="coluna"
            )

        rows: list[dict[str, Any]] = []
        for column in self.df.columns:
            counts = self.df[column].value_counts(dropna=True)
            rows.append(
                {
                    "coluna": column,
                    "tipo": str(self.df[column].dtype),
                    "preenchidos": int(self.df[column].notna().sum()),
                    "valores_unicos": int(self.df[column].nunique(dropna=True)),
                    "valor_mais_frequente": json_value(counts.index[0]) if not counts.empty else None,
                    "frequencia": int(counts.iloc[0]) if not counts.empty else 0,
                }
            )
        pd.DataFrame(rows).to_csv(self.output_path / "column_summary.csv", index=False)

    def analyze_archive(self) -> dict[str, Any]:
        """Analyze years, media types, formats, and archive sources."""
        result: dict[str, Any] = {}
        specifications = [
            ("year", ("ano", "desfile"), "02_year_distribution", "Itens por ano do desfile"),
            ("file_type", ("tipo", "arquivo"), "03_file_type_distribution", "Itens por tipo de arquivo"),
            ("file_format", ("formato", "arquivo"), "04_file_format_distribution", "Itens por formato"),
            ("source", ("fonte",), None, None),
        ]

        for key, terms, chart_name, title in specifications:
            column = find_column(self.df.columns, *terms)
            if not column:
                continue

            series = self.df[column].dropna()
            if key == "year":
                series = pd.to_numeric(series, errors="coerce").dropna().astype(int)
                frequency_counts = series.value_counts()
                counts = frequency_counts.sort_index()
            else:
                counts = series.astype(str).str.strip().replace("", np.nan).dropna().value_counts()
                frequency_counts = counts

            counts.rename_axis(column).rename("quantidade").to_csv(
                self.output_path / f"{key}_counts.csv"
            )
            result[key] = {
                "column": column,
                "non_null": int(series.notna().sum()),
                "unique": int(series.nunique()),
                "most_frequent": (
                    json_value(frequency_counts.index[0]) if not frequency_counts.empty else None
                ),
                "most_frequent_count": (
                    int(frequency_counts.iloc[0]) if not frequency_counts.empty else 0
                ),
            }
            if key == "year" and not series.empty:
                result[key]["first_year"] = int(series.min())
                result[key]["last_year"] = int(series.max())

            if chart_name and not counts.empty:
                shown = counts if key == "year" else counts.head(15).sort_values()
                width = max(11, len(shown) * 0.28) if key == "year" else 11
                figure, axis = plt.subplots(figsize=(width, 6))
                if key == "year":
                    axis.bar(shown.index.astype(str), shown.values, color="#4c72b0")
                    axis.tick_params(axis="x", rotation=90)
                else:
                    axis.barh(shown.index.astype(str), shown.values, color="#4c72b0")
                if key == "year":
                    axis.set(
                        title=title,
                        xlabel="Parade year",
                        ylabel="Number of items",
                    )
                else:
                    axis.set(
                        title=title,
                        xlabel="Number of items",
                        ylabel="File type" if key == "file_type" else "File format",
                    )
                self._save_figure(figure, chart_name)

        return result

    def analyze_media(self) -> dict[str, Any]:
        """Inspect local media files and calculate image and duration statistics."""
        type_column = find_column(self.df.columns, "tipo", "arquivo")
        path_column = find_column(self.df.columns, "localpath")
        if not type_column or not path_column:
            return {"available": False}

        normalized_types = self.df[type_column].fillna("não informado").astype(str).map(normalize_name)
        image_mask = normalized_types.eq("imagem")
        image_rows = self.df.loc[image_mask].copy()
        id_column = find_column(self.df.columns, "id")
        item_column = find_column(self.df.columns, "item")

        inventory: list[dict[str, Any]] = []
        for index, row in image_rows.iterrows():
            raw_path = row[path_column]
            record: dict[str, Any] = {
                "indice": int(index),
                "id": json_value(row[id_column]) if id_column else None,
                "item": json_value(row[item_column]) if item_column else None,
                "caminho": json_value(raw_path),
                "arquivo_encontrado": False,
                "imagem_legivel": False,
                "largura_px": None,
                "altura_px": None,
                "megapixels": None,
                "orientacao": None,
                "formato_detectado": None,
                "tamanho_kib": None,
                "erro": None,
            }
            if pd.isna(raw_path):
                record["erro"] = "caminho ausente"
                inventory.append(record)
                continue

            path = self._resolve_local_path(str(raw_path))
            if not path.is_file():
                record["erro"] = "arquivo não encontrado"
                inventory.append(record)
                continue

            record["arquivo_encontrado"] = True
            record["tamanho_kib"] = round(path.stat().st_size / 1024, 2)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                    with Image.open(path) as image:
                        width, height = image.size
                        detected_format = image.format
                record.update(
                    {
                        "imagem_legivel": True,
                        "largura_px": width,
                        "altura_px": height,
                        "megapixels": round(width * height / 1_000_000, 4),
                        "orientacao": (
                            "paisagem" if width > height else "retrato" if height > width else "quadrada"
                        ),
                        "formato_detectado": detected_format,
                    }
                )
            except (OSError, ValueError) as error:
                record["erro"] = str(error)
            inventory.append(record)

        inventory_df = pd.DataFrame(inventory)
        inventory_df.to_csv(self.output_path / "image_inventory.csv", index=False)
        readable = inventory_df[inventory_df["imagem_legivel"]].copy()
        self.valid_image_indices = readable["indice"].astype(int).tolist()
        unique_readable = readable.drop_duplicates("caminho") if not readable.empty else readable

        if not unique_readable.empty:
            figure, axis = plt.subplots(figsize=(10, 7))
            axis.scatter(
                unique_readable["largura_px"],
                unique_readable["altura_px"],
                s=np.clip(unique_readable["megapixels"] * 4, 8, 70),
                alpha=0.45,
                color="#4c72b0",
                edgecolors="none",
            )
            axis.set(
                title="Dimensões das imagens do acervo",
                xlabel="Width (px)",
                ylabel="Height (px)",
            )
            self._save_figure(figure, "06_image_dimensions")

            resolution_bins = pd.cut(
                unique_readable["megapixels"],
                bins=[-np.inf, 1, 3, 8, 12, np.inf],
                labels=["< 1 MP", "1–3 MP", "3–8 MP", "8–12 MP", "> 12 MP"],
                right=False,
            ).value_counts(sort=False)
            resolution_bins.rename_axis("faixa").rename("quantidade").to_csv(
                self.output_path / "image_resolution_ranges.csv"
            )
            figure, axis = plt.subplots(figsize=(9, 5))
            axis.bar(resolution_bins.index.astype(str), resolution_bins.values, color="#8172b3")
            axis.set(
                title="Imagens por faixa de resolução",
                xlabel="Resolution",
                ylabel="Number of images",
            )
            self._save_figure(figure, "07_image_resolution_ranges")

        path_records: list[dict[str, Any]] = []
        for index, raw_path in self.df[path_column].items():
            if pd.isna(raw_path):
                continue
            path = self._resolve_local_path(str(raw_path))
            if path.is_file():
                path_records.append(
                    {
                        "tipo": normalized_types.loc[index],
                        "caminho": str(path),
                        "tamanho_kib": path.stat().st_size / 1024,
                    }
                )
        files_df = pd.DataFrame(path_records).drop_duplicates("caminho")
        size_by_type = pd.DataFrame()
        if not files_df.empty:
            size_by_type = (
                files_df.groupby("tipo", as_index=False)
                .agg(
                    arquivos=("caminho", "count"),
                    tamanho_total_mib=("tamanho_kib", lambda values: values.sum() / 1024),
                    tamanho_medio_kib=("tamanho_kib", "mean"),
                    tamanho_mediano_kib=("tamanho_kib", "median"),
                )
                .sort_values("arquivos", ascending=False)
            )
            size_by_type.to_csv(self.output_path / "file_sizes_by_type.csv", index=False)

            figure, axis = plt.subplots(figsize=(10, 5))
            plotted = size_by_type.sort_values("tamanho_total_mib")
            axis.barh(plotted["tipo"], plotted["tamanho_total_mib"], color="#dd8452")
            axis.set(
                title="Espaço ocupado por tipo de arquivo",
                xlabel="Total size (MiB)",
                ylabel="File type",
            )
            self._save_figure(figure, "08_file_sizes_by_type")

        width_column = find_column(self.df.columns, "largura", "px")
        height_column = find_column(self.df.columns, "altura", "px")
        dpi_column = find_column(self.df.columns, "resolucao", "dpi")
        duration_column = find_column(self.df.columns, "duracao")

        catalog_width = (
            parse_catalog_number(image_rows[width_column], thousands_separator=True)
            if width_column
            else pd.Series(dtype=float)
        )
        catalog_height = (
            parse_catalog_number(image_rows[height_column], thousands_separator=True)
            if height_column
            else pd.Series(dtype=float)
        )
        catalog_dpi = (
            parse_catalog_number(image_rows[dpi_column]) if dpi_column else pd.Series(dtype=float)
        )
        plausible_dpi = catalog_dpi[catalog_dpi.between(1, 1200)]

        durations = pd.Series(dtype=float)
        if duration_column:
            durations = pd.to_timedelta(self.df[duration_column], errors="coerce").dt.total_seconds().dropna()

        orientation_counts = (
            unique_readable["orientacao"].value_counts().astype(int).to_dict()
            if not unique_readable.empty
            else {}
        )
        image_summary = {
            "catalog_items": int(image_mask.sum()),
            "catalog_percent": round(float(image_mask.mean() * 100), 2),
            "unique_local_files": int(inventory_df.loc[inventory_df["arquivo_encontrado"], "caminho"].nunique()),
            "readable_items": int(len(readable)),
            "unreadable_or_missing_items": int(len(inventory_df) - len(readable)),
            "unique_readable_files": int(unique_readable["caminho"].nunique()) if not readable.empty else 0,
            "average_width_px": round(float(unique_readable["largura_px"].mean()), 2) if not unique_readable.empty else None,
            "average_height_px": round(float(unique_readable["altura_px"].mean()), 2) if not unique_readable.empty else None,
            "median_width_px": round(float(unique_readable["largura_px"].median()), 2) if not unique_readable.empty else None,
            "median_height_px": round(float(unique_readable["altura_px"].median()), 2) if not unique_readable.empty else None,
            "average_megapixels": round(float(unique_readable["megapixels"].mean()), 2) if not unique_readable.empty else None,
            "median_megapixels": round(float(unique_readable["megapixels"].median()), 2) if not unique_readable.empty else None,
            "smallest_megapixels": round(float(unique_readable["megapixels"].min()), 4) if not unique_readable.empty else None,
            "largest_megapixels": round(float(unique_readable["megapixels"].max()), 4) if not unique_readable.empty else None,
            "orientation": orientation_counts,
            "average_file_size_kib": round(float(unique_readable["tamanho_kib"].mean()), 2) if not unique_readable.empty else None,
            "median_file_size_kib": round(float(unique_readable["tamanho_kib"].median()), 2) if not unique_readable.empty else None,
            "total_file_size_mib": round(float(unique_readable["tamanho_kib"].sum() / 1024), 2) if not unique_readable.empty else None,
            "catalog_width_coverage_percent": round(float(catalog_width.notna().mean() * 100), 2),
            "catalog_height_coverage_percent": round(float(catalog_height.notna().mean() * 100), 2),
            "catalog_dpi_coverage_percent": round(float(catalog_dpi.notna().mean() * 100), 2),
            "average_catalog_dpi": round(float(plausible_dpi.mean()), 2) if not plausible_dpi.empty else None,
            "median_catalog_dpi": round(float(plausible_dpi.median()), 2) if not plausible_dpi.empty else None,
            "suspicious_catalog_dpi_values": int((catalog_dpi > 1200).sum()),
        }

        return {
            "available": True,
            "items_by_type": self.df[type_column].value_counts().astype(int).to_dict(),
            "images": image_summary,
            "files": {
                "unique_files_found": int(len(files_df)),
                "total_size_mib": round(float(files_df["tamanho_kib"].sum() / 1024), 2)
                if not files_df.empty
                else 0,
                "by_type": size_by_type.round(2).to_dict("records") if not size_by_type.empty else [],
            },
            "duration": {
                "items_with_duration": int(len(durations)),
                "total_seconds": round(float(durations.sum()), 2),
                "total_hours": round(float(durations.sum() / 3600), 2),
                "average_seconds": round(float(durations.mean()), 2) if not durations.empty else None,
                "median_seconds": round(float(durations.median()), 2) if not durations.empty else None,
            },
        }

    def analyze_keywords(self) -> dict[str, Any]:
        """Analyze keyword frequency and pairwise co-occurrence."""
        if not self.keyword_columns:
            return {"detected": 0}

        keywords = self.df[self.keyword_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        keywords = keywords.clip(0, 1).astype(int)
        frequencies = keywords.sum().sort_values(ascending=False)
        frequencies.rename_axis("palavra_chave").rename("quantidade").to_csv(
            self.output_path / "keyword_frequencies.csv"
        )

        cooccurrence = keywords.T.dot(keywords)
        np.fill_diagonal(cooccurrence.values, 0)
        pairs: list[dict[str, Any]] = []
        for position, first in enumerate(self.keyword_columns):
            for second in self.keyword_columns[position + 1 :]:
                count = int(cooccurrence.loc[first, second])
                if count:
                    pairs.append({"palavra_chave_1": first, "palavra_chave_2": second, "quantidade": count})
        pair_table = pd.DataFrame(
            pairs,
            columns=["palavra_chave_1", "palavra_chave_2", "quantidade"],
        ).sort_values("quantidade", ascending=False)
        pair_table.to_csv(self.output_path / "keyword_cooccurrences.csv", index=False)

        figure, axis = plt.subplots(figsize=(11, max(6, len(frequencies) * 0.32)))
        shown = frequencies.sort_values()
        axis.barh(shown.index.str.replace("\n", " ", regex=False), shown.values, color="#55a868")
        axis.set(
            title="Frequência das palavras-chave",
            xlabel="Number of items",
            ylabel="Keywords",
        )
        self._save_figure(figure, "05_keyword_frequencies")

        labels_per_item = keywords.sum(axis=1)
        result = {
            "detected": len(self.keyword_columns),
            "items_without_keywords": int((labels_per_item == 0).sum()),
            "average_keywords_per_item": round(float(labels_per_item.mean()), 2),
            "most_frequent": frequencies.index[0] if not frequencies.empty else None,
            "most_frequent_count": int(frequencies.iloc[0]) if not frequencies.empty else 0,
        }

        valid_indices = self.df.index.intersection(self.valid_image_indices)
        if valid_indices.empty:
            result["valid_images"] = {"total": 0}
            return result

        valid_keywords = keywords.loc[valid_indices]
        valid_frequencies = valid_keywords.sum().sort_values(ascending=False)
        valid_distribution = pd.DataFrame(
            {
                "palavra_chave": valid_frequencies.index,
                "quantidade": valid_frequencies.values.astype(int),
                "percentual_imagens_validas": (
                    valid_frequencies.values / len(valid_keywords) * 100
                ).round(2),
            }
        )
        valid_distribution.to_csv(
            self.output_path / "valid_image_keyword_distribution.csv", index=False
        )

        valid_label_counts = valid_keywords.sum(axis=1).astype(int)
        count_distribution = valid_label_counts.value_counts().sort_index()
        pd.DataFrame(
            {
                "numero_de_chaves": count_distribution.index,
                "quantidade_de_imagens": count_distribution.values,
                "percentual_imagens_validas": (
                    count_distribution.values / len(valid_keywords) * 100
                ).round(2),
            }
        ).to_csv(self.output_path / "valid_image_keyword_count_distribution.csv", index=False)

        shown = valid_distribution.sort_values("quantidade")
        figure, axis = plt.subplots(figsize=(11, max(6, len(shown) * 0.32)))
        bars = axis.barh(
            shown["palavra_chave"].str.replace("\n", " ", regex=False),
            shown["percentual_imagens_validas"],
            color="#55a868",
        )
        axis.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
        axis.set(
            title=f"Distribuição das palavras-chave em {len(valid_keywords)} imagens válidas",
            xlabel="Valid images with keyword (%)",
            ylabel="Keywords",
        )
        axis.set_xlim(0, max(shown["percentual_imagens_validas"].max() * 1.18, 1))
        self._save_figure(figure, "09_valid_image_keyword_distribution")

        figure, axis = plt.subplots(figsize=(9, 5))
        bars = axis.bar(
            count_distribution.index.astype(str),
            count_distribution.values,
            color="#64b5cd",
        )
        axis.bar_label(bars, padding=3, fontsize=9)
        axis.set(
            title="Quantidade de palavras-chave por imagem válida",
            xlabel="Number of keywords",
            ylabel="Number of images",
        )
        self._save_figure(figure, "10_valid_image_keyword_count_distribution")

        result["valid_images"] = {
            "total": int(len(valid_keywords)),
            "without_keywords": int((valid_label_counts == 0).sum()),
            "average_keywords_per_image": round(float(valid_label_counts.mean()), 2),
            "median_keywords_per_image": round(float(valid_label_counts.median()), 2),
            "distribution": valid_distribution.to_dict("records"),
        }
        return result

    def run(self) -> Path:
        """Run every analysis and return the generated JSON report path."""
        quality = self.analyze_quality()
        self.analyze_descriptive_statistics()
        archive = self.analyze_archive()
        media = self.analyze_media()
        keywords = self.analyze_keywords()

        report = {
            "dataset": str(self.dataset_path),
            "generated_at": pd.Timestamp.now().isoformat(),
            "quality": quality,
            "archive": archive,
            "media": media,
            "keywords": keywords,
        }
        report_path = self.output_path / "dataset_analysis_report.json"
        with report_path.open("w", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, indent=2, default=json_value)

        print(f"Dataset: {self.dataset_path}")
        print(f"Dimensões: {quality['rows']} linhas × {quality['columns']} colunas")
        print(f"Linhas duplicadas: {quality['duplicate_rows']}")
        if media.get("available"):
            images = media["images"]
            print(f"Imagens: {images['catalog_items']} itens ({images['catalog_percent']}%)")
            print(
                "Resolução média: "
                f"{images['average_width_px']:.0f} × {images['average_height_px']:.0f} px "
                f"({images['average_megapixels']:.2f} MP)"
            )
        print(f"Palavras-chave detectadas: {keywords['detected']}")
        print(f"Resultados: {self.output_path}")
        return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analisa o dataset do acervo Suvaco de Cristo.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"CSV a analisar (padrão: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Diretório de saída (padrão: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DatasetAnalyzer(args.dataset, args.output).run()


if __name__ == "__main__":
    main()
