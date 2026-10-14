"""Dataset processing module.

This module handles data processing tasks including extracting hyperlinks from Excel,
processing category keywords, and merging datasets. It integrates hyperlink extraction
functionality previously in extract_hyperlinks.py.
"""

import pandas as pd
from pathlib import Path
from openpyxl import load_workbook
from openpyxl.utils.cell import column_index_from_string


def extract_hyperlinks_from_excel(xlsx_path: str, sheet_name: str, col_letter: str,
                                   header_rows: int = 2) -> pd.DataFrame:
    """Extract hyperlinks from Excel cells and create DataFrame with links.

    Args:
        xlsx_path: Path to Excel file.
        sheet_name: Name of the sheet.
        col_letter: Column letter containing hyperlinks (e.g., 'B').
        header_rows: Number of header rows before data. Defaults to 2.

    Returns:
        pd.DataFrame: DataFrame with 'Hyperlink' column added.
    """
    # Read Excel data with pandas to get structured DataFrame
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=header_rows)

    # Open with openpyxl to access hyperlink metadata
    # (pandas doesn't preserve hyperlinks, so we need openpyxl)
    wb = load_workbook(xlsx_path, data_only=False)
    ws = wb[sheet_name]

    # Extract exactly len(df) hyperlinks from rows after header
    n = len(df)
    # Calculate starting row: +1 for Excel's 1-indexing, +header_rows to skip headers
    start_row = header_rows + 2  # +1 for 1-indexed, +1 to skip header row
    col_idx = column_index_from_string(col_letter)

    # Collect hyperlinks from the specified column
    links = []
    for r in range(start_row, start_row + n):
        cell = ws.cell(row=r, column=col_idx)
        # Extract hyperlink target if it exists, otherwise None
        if cell.hyperlink is not None:
            links.append(cell.hyperlink.target)
        else:
            links.append(None)

    # Add hyperlinks as a new column to DataFrame
    df["Hyperlink"] = links

    return df


def process_palavras_chave(csv_path: str) -> pd.DataFrame:
    """Process keywords CSV: convert 'X' to 1 and empty to 0.

    Args:
        csv_path: Path to keywords CSV file.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    print(f"\nProcessing keywords from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Columns to ignore (not categories)
    ignore_cols = ['Id', 'Item']

    # Process each category column (excluding metadata columns)
    for col in df.columns:
        if col not in ignore_cols:
            # Convert 'X' markers to 1, everything else to 0 (binary encoding)
            df[col] = df[col].apply(lambda x: 1 if str(x).strip().upper() == 'X' else 0)

    print(f"✓ Processed {len(df.columns) - len(ignore_cols)} category columns")

    return df


def merge_dataframes(catalogacao_df: pd.DataFrame, palavras_chave_df: pd.DataFrame) -> pd.DataFrame:
    """Merge catalog and keywords tables.

    Args:
        catalogacao_df: Catalog DataFrame.
        palavras_chave_df: Processed keywords DataFrame.

    Returns:
        pd.DataFrame: Merged DataFrame.

    Raises:
        ValueError: If no common column ('Id' or 'Item') is found for merging.
    """
    print("\nMerging tables...")

    # Merge using 'Id' column if available, otherwise fall back to 'Item'
    if 'Id' in catalogacao_df.columns and 'Id' in palavras_chave_df.columns:
        # Left join preserves all catalog entries
        merged_df = pd.merge(catalogacao_df, palavras_chave_df, on='Id', how='left', suffixes=('', '_palavras'))
        print(f"✓ Tables merged using 'Id' column")
    elif 'Item' in catalogacao_df.columns and 'Item' in palavras_chave_df.columns:
        # Alternative: merge by Item name
        merged_df = pd.merge(catalogacao_df, palavras_chave_df, on='Item', how='left', suffixes=('', '_palavras'))
        print(f"✓ Tables merged using 'Item' column")
    else:
        raise ValueError("Could not find common column for merge (Id or Item)")

    # Clean up: remove duplicate columns created by merge
    if 'Item_palavras' in merged_df.columns:
        merged_df = merged_df.drop(columns=['Item_palavras'])

    print(f"✓ Final DataFrame: {merged_df.shape[0]} rows x {merged_df.shape[1]} columns")

    return merged_df


if __name__ == "__main__":
    # Paths
    xlsx_path = "dataset/Descrição do acervo.xlsx"
    output_dir = Path("dataset")

    print("="*80)
    print("DATASET DATA PROCESSING")
    print("="*80)

    # 1. Extract hyperlinks from Excel
    print("\n1. Extracting hyperlinks from Excel...")
    catalogacao_df = extract_hyperlinks_from_excel(
        xlsx_path=xlsx_path,
        sheet_name="Catalogação",
        col_letter="B",  # "Item" column
        header_rows=2    # First 2 rows are not header
    )

    # Extract keywords data
    palavras_chave_df = pd.read_excel(
        xlsx_path,
        sheet_name="catalogação das palavras-chave ",
        header=2
    )

    # Save intermediate CSVs
    catalogacao_csv = output_dir / "catalogacao_with_hyperlinks.csv"
    palavras_chave_csv = output_dir / "palavras_chave.csv"

    catalogacao_df.to_csv(catalogacao_csv, index=False)
    palavras_chave_df.to_csv(palavras_chave_csv, index=False)

    print(f"\n✓ Saved files:")
    print(f"  - {catalogacao_csv} ({len(catalogacao_df)} rows)")
    print(f"  - {palavras_chave_csv} ({len(palavras_chave_df)} rows)")

    # Hyperlink statistics
    total_links = catalogacao_df['Hyperlink'].notna().sum()
    missing_links = catalogacao_df['Hyperlink'].isna().sum()

    print(f"\n✓ Extracted hyperlinks:")
    print(f"  - With hyperlink: {total_links}")
    print(f"  - Without hyperlink: {missing_links}")

    # 2. Process keywords (convert X to 1, empty to 0)
    palavras_chave_df = process_palavras_chave(palavras_chave_csv)
    palavras_chave_processed_csv = "dataset/palavras_chave_processed.csv"
    palavras_chave_df.to_csv(palavras_chave_processed_csv, index=False)
    print(f"✓ Processed keywords saved to: {palavras_chave_processed_csv}")

    # 3. Load catalog with local paths (if exists)
    catalogacao_with_paths_csv = "dataset/catalogacao_with_local_paths.csv"
    if Path(catalogacao_with_paths_csv).exists():
        print(f"\n✓ Loading catalog with local paths from: {catalogacao_with_paths_csv}")
        catalogacao_df = pd.read_csv(catalogacao_with_paths_csv)
    else:
        print(f"\n⚠ Warning: {catalogacao_with_paths_csv} not found. Run download.py first.")
        print("  Continuing with catalog without local paths...")

    # 4. Merge tables
    merged_df = merge_dataframes(catalogacao_df, palavras_chave_df)

    # 5. Filter items where "Repositório = 0 para excluir item" != 0
    repo_col = "Repositório = 0 para excluir item"
    if repo_col in merged_df.columns:
        initial_count = len(merged_df)
        merged_df = merged_df[merged_df[repo_col] != 0].copy()
        removed_count = initial_count - len(merged_df)
        print(f"\n✓ Removed {removed_count} items where '{repo_col}' = 0")
        print(f"✓ Dataset after filtering: {len(merged_df)} items")
    else:
        print(f"\n⚠ Warning: Column '{repo_col}' not found - skipping filtering")

    # 6. Filter only image files
    print("\nFiltering only image files...")
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}

    def is_image_file(path):
        """Check if file is an image.

        Args:
            path: File path.

        Returns:
            bool: True if file is an image, False otherwise.
        """
        if pd.isna(path):
            return False
        return Path(path).suffix.lower() in image_extensions

    initial_count = len(merged_df)
    merged_df['is_image'] = merged_df['LocalPath'].apply(is_image_file)
    merged_df = merged_df[merged_df['is_image']].copy()
    merged_df = merged_df.drop(columns=['is_image'])

    removed_count = initial_count - len(merged_df)
    print(f"✓ Removed {removed_count} non-image files")
    print(f"✓ Final dataset (images only): {len(merged_df)} items")

    # Save final dataset
    final_csv = "dataset/dataset_completo.csv"
    merged_df.to_csv(final_csv, index=False)
    print(f"\n✓ Complete dataset saved to: {final_csv}")

    # Final statistics
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total items (images): {len(merged_df)}")
    print(f"Downloaded files: {merged_df['LocalPath'].notna().sum()}")
    print(f"Total columns: {len(merged_df.columns)}")

    # Count categories
    ignore_cols = ['Id', 'Item', 'LocalPath', 'Hyperlink']
    category_cols = [col for col in palavras_chave_df.columns if col not in ignore_cols]
    print(f"Keyword categories: {len(category_cols)}")

    print("\n✓ Processing complete!")
