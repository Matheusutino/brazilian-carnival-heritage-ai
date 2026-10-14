"""Dataset Preparation and Processing Module.

This module provides functionality for preparing and processing the Bloco Suvaco de Cristo
dataset. It handles data extraction from Excel files, Google Drive downloads, category
processing, and dataset merging operations.

The module includes:
    - Excel hyperlink extraction
    - Google Drive file downloads with authentication
    - Category keyword processing (converting 'X' markers to binary)
    - Dataset merging and filtering
    - Image file validation

Main components:
    - download.py: Google Drive authentication and parallel file downloads
    - process.py: Data extraction, processing, and merging operations

Example usage:
    >>> # Process Excel data and extract hyperlinks
    >>> from src.core.dataset.process import extract_hyperlinks_from_excel
    >>> df = extract_hyperlinks_from_excel("data.xlsx", "Sheet1", "B", header_rows=2)

    >>> # Download files from Google Drive
    >>> from src.core.dataset.download import download_all_files
    >>> df = download_all_files("catalog.csv", "downloads/", max_workers=10)
"""

__version__ = "1.0.0"
