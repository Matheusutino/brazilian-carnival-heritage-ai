# Dataset Module

The `src/core/dataset` module provides dataset preparation and processing functionality for the Bloco Suvaco de Cristo archive project. It handles Excel data extraction, Google Drive file downloads, category processing, and dataset assembly.

## Overview

This module orchestrates the complete dataset preparation pipeline:

1. **Excel Processing** - Extract hyperlinks and structured data from Excel files
2. **Google Drive Integration** - Download files using service account authentication
3. **Category Processing** - Convert keyword markers to binary labels
4. **Dataset Assembly** - Merge tables and filter valid entries
5. **Validation** - Filter image files and validate data integrity

## Files

### `download.py`

Google Drive file downloader with parallel processing and authentication.

**Key Features:**
- Service account authentication for Google Drive API
- Parallel multi-threaded downloads
- Support for multiple Google Drive URL formats
- Export Google Docs/Sheets/Slides to standard formats (PDF, XLSX)
- Smart file ID extraction from various URL types
- Skip already downloaded files
- Image-only filtering
- Progress tracking with success/failure counts

**Main Functions:**

#### `authenticate_google_drive()`
Authenticate with Google Drive API using service account credentials.

**Returns:** Google Drive API service object

**Environment Variables:**
- `GOOGLE_SERVICE_ACCOUNT_JSON`: Path to credentials file (default: `credentials.json`)

#### `extract_file_id_from_url(url: str) -> str`
Extract file ID from various Google Drive URL formats.

**Supported URL Formats:**
- `https://drive.google.com/file/d/{FILE_ID}/view`
- `https://docs.google.com/document/d/{FILE_ID}/edit`
- `https://docs.google.com/spreadsheets/d/{FILE_ID}/edit`
- `https://docs.google.com/presentation/d/{FILE_ID}/edit`
- `https://drive.google.com/open?id={FILE_ID}`

#### `download_all_files(csv_path, output_dir, max_workers=10, skip_existing=True)`
Download all files from Google Drive based on CSV with hyperlinks.

**Args:**
- `csv_path`: Path to CSV file containing hyperlinks
- `output_dir`: Directory to save downloaded files
- `max_workers`: Maximum number of parallel threads (default: 10)
- `skip_existing`: Skip files that already exist (default: True)

**Returns:** DataFrame with added `LocalPath` column

**Usage Example:**
```python
from src.core.dataset.download import download_all_files

# Download files from catalog
df = download_all_files(
    csv_path='dataset/catalogacao_with_hyperlinks.csv',
    output_dir='dataset/downloaded',
    max_workers=10,
    skip_existing=True
)

# Save catalog with local paths
df.to_csv('dataset/catalogacao_with_local_paths.csv', index=False)
```

### `process.py`

Data processing utilities for Excel extraction, category encoding, and dataset merging.

**Key Features:**
- Extract hyperlinks embedded in Excel cells using openpyxl
- Convert category markers ('X') to binary labels (0/1)
- Merge multiple data sources
- Filter excluded items
- Image file validation

**Main Functions:**

#### `extract_hyperlinks_from_excel(xlsx_path, sheet_name, col_letter, header_rows=2)`
Extract hyperlinks from Excel cells and create DataFrame.

**Args:**
- `xlsx_path`: Path to Excel file
- `sheet_name`: Name of the sheet
- `col_letter`: Column letter containing hyperlinks (e.g., 'B')
- `header_rows`: Number of header rows before data (default: 2)

**Returns:** DataFrame with 'Hyperlink' column added

#### `process_palavras_chave(csv_path: str)`
Process keywords CSV: convert 'X' to 1 and empty to 0.

**Args:**
- `csv_path`: Path to keywords CSV file

**Returns:** Processed DataFrame with binary category values

#### `merge_dataframes(catalogacao_df, palavras_chave_df)`
Merge catalog and keywords tables.

**Args:**
- `catalogacao_df`: Catalog DataFrame
- `palavras_chave_df`: Processed keywords DataFrame

**Returns:** Merged DataFrame

**Usage Example:**
```python
from src.core.dataset.process import (
    extract_hyperlinks_from_excel,
    process_palavras_chave,
    merge_dataframes
)

# Extract hyperlinks from Excel
catalogacao_df = extract_hyperlinks_from_excel(
    xlsx_path="dataset/Descrição do acervo.xlsx",
    sheet_name="Catalogação",
    col_letter="B",
    header_rows=2
)

# Process keywords (X -> 1, empty -> 0)
palavras_chave_df = process_palavras_chave('dataset/palavras_chave.csv')

# Merge tables
merged_df = merge_dataframes(catalogacao_df, palavras_chave_df)

# Filter out excluded items
repo_col = "Repositório = 0 para excluir item"
if repo_col in merged_df.columns:
    merged_df = merged_df[merged_df[repo_col] != 0]

# Save final dataset
merged_df.to_csv('dataset/dataset_completo.csv', index=False)
```

### `__init__.py`

Package initialization file with module documentation.

## Google Drive Setup

Before using the download functionality, you need to configure Google Drive API access. See `GOOGLE_DRIVE_SETUP.md` for detailed instructions.

**Quick Setup:**
1. Create a Google Cloud project
2. Enable Google Drive API
3. Create a service account
4. Download credentials as `credentials.json`
5. Share Google Drive folder with service account email
6. Place `credentials.json` in project root or set `GOOGLE_SERVICE_ACCOUNT_JSON` environment variable

## Supported File Types

### Download Module

The downloader handles various file types:

**Regular Files:**
- Direct download for images, PDFs, videos, etc.

**Google Workspace Files (auto-converted):**
- Google Docs → PDF
- Google Sheets → XLSX
- Google Presentations → PDF

### Image Filtering

Only image files are processed. Supported extensions:
- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- GIF (`.gif`)
- BMP (`.bmp`)
- WebP (`.webp`)
- TIFF (`.tiff`, `.tif`)

## Generated Files

The module generates the following files in the `dataset/` directory:

| File | Description |
|------|-------------|
| `catalogacao_with_hyperlinks.csv` | Catalog with extracted hyperlink column |
| `palavras_chave.csv` | Original keywords data |
| `palavras_chave_processed.csv` | Keywords with binary encoding (X→1, empty→0) |
| `catalogacao_with_local_paths.csv` | Catalog with local file paths after download |
| `dataset_completo.csv` | **Final merged and filtered dataset** |
| `downloaded/` | Directory containing all downloaded files |

## Final Dataset Structure

The `dataset_completo.csv` file contains:

**Metadata Columns:**
- `Id`: Unique identifier
- `Item`: Item name (filename)
- `Hyperlink`: Google Drive URL
- `LocalPath`: Local file path after download
- `Ano do desfile`: Carnival year
- `Tipo do arquivo`: File type (photo, video, document)
- `Repositório = 0 para excluir item`: Exclusion flag

**Category Columns:**
- Binary values (0 or 1) for each keyword category
- 0 = category not present
- 1 = category present (marked with 'X' in original)

## Complete Pipeline Example

Running the full dataset preparation:

```python
from pathlib import Path
from src.core.dataset.process import (
    extract_hyperlinks_from_excel,
    process_palavras_chave,
    merge_dataframes
)
from src.core.dataset.download import download_all_files
import pandas as pd

# Configuration
xlsx_path = "dataset/Descrição do acervo.xlsx"
output_dir = Path("dataset")

print("Step 1: Extract hyperlinks from Excel...")
catalogacao_df = extract_hyperlinks_from_excel(
    xlsx_path=xlsx_path,
    sheet_name="Catalogação",
    col_letter="B",
    header_rows=2
)

palavras_chave_df = pd.read_excel(
    xlsx_path,
    sheet_name="catalogação das palavras-chave ",
    header=2
)

# Save intermediate files
catalogacao_df.to_csv(output_dir / "catalogacao_with_hyperlinks.csv", index=False)
palavras_chave_df.to_csv(output_dir / "palavras_chave.csv", index=False)

print("Step 2: Download files from Google Drive...")
catalogacao_df = download_all_files(
    csv_path=output_dir / "catalogacao_with_hyperlinks.csv",
    output_dir=output_dir / "downloaded",
    max_workers=10,
    skip_existing=True
)

# Save catalog with paths
catalogacao_df.to_csv(output_dir / "catalogacao_with_local_paths.csv", index=False)

print("Step 3: Process keywords...")
palavras_chave_df = process_palavras_chave(output_dir / "palavras_chave.csv")
palavras_chave_df.to_csv(output_dir / "palavras_chave_processed.csv", index=False)

print("Step 4: Merge and filter dataset...")
merged_df = merge_dataframes(catalogacao_df, palavras_chave_df)

# Filter excluded items
repo_col = "Repositório = 0 para excluir item"
if repo_col in merged_df.columns:
    merged_df = merged_df[merged_df[repo_col] != 0]

# Filter only images
image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
merged_df = merged_df[merged_df['LocalPath'].apply(
    lambda p: Path(str(p)).suffix.lower() in image_extensions if pd.notna(p) else False
)]

# Save final dataset
merged_df.to_csv(output_dir / "dataset_completo.csv", index=False)
print(f"✓ Complete! Final dataset: {len(merged_df)} images")
```

## Command-Line Execution

You can also run the processing modules directly:

```bash
# Process Excel and merge data
python -m src.core.dataset.process

# Download files from Google Drive
python -m src.core.dataset.download
```

## Error Handling

### Common Issues

**Google Drive Authentication Error:**
```
ValueError: Google Drive credentials not found at: credentials.json
```
**Solution:** Follow setup instructions in `GOOGLE_DRIVE_SETUP.md`

**Permission Denied on Download:**
```
HTTP error while downloading: 403 Forbidden
```
**Solution:** Share the Google Drive folder with your service account email

**Missing Hyperlinks:**
- Some catalog entries may not have hyperlinks
- These will have `None` in the `Hyperlink` column
- No file will be downloaded for these entries

**Non-Image Files:**
- Automatically filtered during processing
- Only image files are included in final dataset

## Performance Considerations

### Download Performance

**Parallel Downloads:**
- Default: 10 concurrent threads
- Adjust `max_workers` based on network and API limits
- Higher values may hit rate limits (use 5-15 for optimal performance)

**Skip Existing:**
- Enable `skip_existing=True` to resume interrupted downloads
- Checks file existence before downloading
- Significantly faster for incremental updates

### Memory Usage

**Large Datasets:**
- Excel processing loads entire file into memory
- For very large datasets (>100k rows), consider chunked processing
- Downloads are streamed to disk (low memory footprint)

## Integration with Project

This module is the foundation of the Bloco Suvaco de Cristo classification pipeline:

1. **Dataset Preparation** → This module
2. **Train/Test Split** → `src/scripts/preparation/create_train_test_split.py`
3. **Feature Extraction** → `src/core/features/`
4. **Classification** → `src/core/classification/`

The output `dataset_completo.csv` is used by downstream modules for:
- Creating stratified train/test splits
- Loading images for feature extraction
- Training multilabel classifiers
- Generating captions and metadata
