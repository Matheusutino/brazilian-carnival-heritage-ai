"""Google Drive file download module.

This module provides functionality to download files from Google Drive,
including support for exporting Google Docs, Sheets, and Presentations.
It uses service account authentication and supports parallel downloads.
"""

import os
import io
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

# Load environment variables
load_dotenv()


def authenticate_google_drive():
    """Authenticate with Google Drive API using service account.

    Returns:
        Resource: Google Drive API service object.

    Raises:
        ValueError: If credentials file is not found.
    """
    credentials_path = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON', 'credentials.json')

    if not os.path.exists(credentials_path):
        raise ValueError(
            f"Google Drive credentials not found at: {credentials_path}\n"
            f"See GOOGLE_DRIVE_SETUP.md for instructions."
        )

    credentials = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )

    service = build('drive', 'v3', credentials=credentials)
    return service


def extract_file_id_from_url(url: str) -> str:
    """Extract file ID from various Google URL formats.

    Args:
        url: Google Drive URL.

    Returns:
        str: Extracted file ID, or None if URL is invalid.
    """
    if not url or pd.isna(url):
        return None

    url = str(url)

    # Different URL formats supported by Google Drive
    # Format: https://drive.google.com/file/d/{FILE_ID}/view
    if '/file/d/' in url:
        return url.split('/file/d/')[1].split('/')[0]
    # Format: https://docs.google.com/document/d/{FILE_ID}/edit
    elif '/document/d/' in url:
        return url.split('/document/d/')[1].split('/')[0]
    # Format: https://docs.google.com/spreadsheets/d/{FILE_ID}/edit
    elif '/spreadsheets/d/' in url:
        return url.split('/spreadsheets/d/')[1].split('/')[0]
    # Format: https://docs.google.com/presentation/d/{FILE_ID}/edit
    elif '/presentation/d/' in url:
        return url.split('/presentation/d/')[1].split('/')[0]
    # Format: https://drive.google.com/open?id={FILE_ID}
    elif 'id=' in url:
        return url.split('id=')[1].split('&')[0]
    # Legacy format
    elif '/open?id=' in url:
        return url.split('/open?id=')[1].split('&')[0]
    else:
        # Assume the entire string is already a file ID
        return url


def get_export_mime_type(mime_type: str) -> tuple:
    """Get export MIME type and extension for Google Docs/Sheets/Slides.

    Args:
        mime_type: MIME type of the file in Google Drive.

    Returns:
        tuple: (export_mime_type, extension) or (None, None) if no export needed.
    """
    export_formats = {
        'application/vnd.google-apps.document': ('application/pdf', '.pdf'),
        'application/vnd.google-apps.spreadsheet': ('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', '.xlsx'),
        'application/vnd.google-apps.presentation': ('application/pdf', '.pdf'),
    }
    return export_formats.get(mime_type, (None, None))


def download_file_from_drive(service, file_id: str, output_path: Path, url: str = None) -> bool:
    """Download file from Google Drive, including Google Docs/Sheets/Slides.

    Args:
        service: Google Drive API service object.
        file_id: File ID in Google Drive.
        output_path: Local path to save the file.
        url: Original URL (for error reporting). Defaults to None.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Get file metadata to determine MIME type
        file_metadata = service.files().get(fileId=file_id, fields='name,mimeType').execute()
        mime_type = file_metadata.get('mimeType', '')
        file_name = file_metadata.get('name', '')

        # Check if it's a Google Doc/Sheet/Slide that needs special export handling
        export_mime_type, extension = get_export_mime_type(mime_type)

        if export_mime_type:
            # Export Google Workspace files (Docs, Sheets, Slides) to standard formats
            request = service.files().export_media(fileId=file_id, mimeType=export_mime_type)

            # Add appropriate extension if not present
            if not output_path.suffix:
                output_path = output_path.with_suffix(extension)
        else:
            # Normal download for regular files (images, PDFs, etc.)
            request = service.files().get_media(fileId=file_id)

        # Download file to memory buffer
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        # Download in chunks until complete
        done = False
        while not done:
            _, done = downloader.next_chunk()

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file from buffer to disk
        with open(output_path, 'wb') as f:
            fh.seek(0)
            f.write(fh.read())

        return True

    except HttpError as e:
        error_msg = f"HTTP error while downloading"
        if url:
            error_msg += f" {url}"
        error_msg += f": {str(e)}"
        print(error_msg)
        return False
    except Exception as e:
        error_msg = f"Error downloading"
        if url:
            error_msg += f" {url}"
        error_msg += f": {str(e)}"
        print(error_msg)
        return False


def download_all_files(csv_path: str, output_dir: str = "dataset/downloaded",
                       max_workers: int = 10, skip_existing: bool = True):
    """Download all files from Google Drive based on CSV with hyperlinks.

    Args:
        csv_path: Path to CSV file containing hyperlinks.
        output_dir: Directory to save downloaded files. Defaults to "dataset/downloaded".
        max_workers: Maximum number of parallel threads. Defaults to 10.
        skip_existing: Skip files that already exist. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with added 'LocalPath' column containing file paths.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print("\nAuthenticating with Google Drive...")
    authenticate_google_drive()  # Test authentication

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Add local path column
    df['LocalPath'] = None

    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}

    # Prepare download tasks
    download_tasks = []
    skipped_non_images = 0

    for idx, row in df.iterrows():
        hyperlink = row.get('Hyperlink')
        if pd.notna(hyperlink):
            # Extract Google Drive file ID from URL
            file_id = extract_file_id_from_url(hyperlink)
            if file_id:
                # Use 'Item' column as filename, or generate fallback name
                item_name = str(row['Item']) if pd.notna(row.get('Item')) else f"item_{idx}"

                # Clean filename by removing invalid characters
                item_name = "".join(c for c in item_name if c.isalnum() or c in (' ', '.', '_', '-'))
                file_path = output_path / item_name

                # Filter: only process image files
                file_ext = Path(item_name).suffix.lower()
                if file_ext not in image_extensions:
                    skipped_non_images += 1
                    continue

                # Skip files that already exist if skip_existing is enabled
                if skip_existing and file_path.exists():
                    df.at[idx, 'LocalPath'] = str(file_path)
                else:
                    # Add to download queue
                    download_tasks.append({
                        'idx': idx,
                        'file_id': file_id,
                        'file_path': file_path,
                        'item_name': item_name,
                        'url': hyperlink
                    })

    if skipped_non_images > 0:
        print(f"✓ Skipped {skipped_non_images} non-image files")

    # Count existing files
    existing_count = df['LocalPath'].notna().sum()
    to_download = len(download_tasks)

    if existing_count > 0:
        print(f"Found {existing_count} already downloaded files")

    if to_download == 0:
        print("All files already downloaded! ✓")
        return df

    print(f"\nDownloading {to_download} files using {max_workers} parallel threads...")

    downloaded_count = 0
    failed_count = 0
    print_lock = Lock()

    def download_task(task):
        """Download a single file - creates its own service instance.

        Args:
            task: Dictionary containing download task information.

        Returns:
            tuple: (task, success) where success is a boolean.
        """
        try:
            service = authenticate_google_drive()
            success = download_file_from_drive(
                service,
                task['file_id'],
                task['file_path'],
                task.get('url')
            )
            return task, success
        except Exception as e:
            print(f"Thread error for {task['item_name']}: {str(e)}")
            return task, False

    # Execute downloads in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(download_task, task): task for task in download_tasks}

        # Process completed downloads
        for future in as_completed(futures):
            task, success = future.result()

            with print_lock:
                if success:
                    df.at[task['idx'], 'LocalPath'] = str(task['file_path'])
                    downloaded_count += 1
                    print(f"[{downloaded_count + failed_count}/{to_download}] ✓ {task['item_name']}")
                else:
                    failed_count += 1
                    print(f"[{downloaded_count + failed_count}/{to_download}] ✗ {task['item_name']}")

    print(f"\nDownload complete! ✓ {downloaded_count} success, ✗ {failed_count} failures, ⊙ {existing_count} already existed")

    return df


if __name__ == "__main__":
    # Paths
    catalogacao_csv = "dataset/catalogacao_with_hyperlinks.csv"
    output_dir = "dataset/downloaded"

    # Download all files from Google Drive
    catalogacao_df = download_all_files(
        csv_path=catalogacao_csv,
        output_dir=output_dir,
        max_workers=10,
        skip_existing=True
    )

    # Save catalog with local paths
    catalogacao_with_paths_csv = "dataset/catalogacao_with_local_paths.csv"
    catalogacao_df.to_csv(catalogacao_with_paths_csv, index=False)
    print(f"\n✓ Catalog with local paths saved to: {catalogacao_with_paths_csv}")
