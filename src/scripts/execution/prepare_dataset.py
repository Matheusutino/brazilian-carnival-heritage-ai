"""Main script for preparing the complete dataset.

This script orchestrates the full dataset preparation pipeline for the
Bloco Suvaco de Cristo project. It executes hyperlink extraction, category
processing, file downloads from Google Drive, and generates the final dataset
with local file paths.
"""

import sys
import subprocess
from pathlib import Path


def run_script(script_path: str, description: str):
    """Execute a Python script and display output in real-time.

    Args:
        script_path (str): Path to the Python script to execute.
        description (str): Description of what the script does.

    Returns:
        bool: True if the script executed successfully, False otherwise.
    """
    print("\n" + "="*80)
    print(f"EXECUTING: {description}")
    print("="*80 + "\n")

    try:
        # Execute the script using the current Python interpreter
        # capture_output=False allows real-time output streaming to console
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,  # Raise CalledProcessError if script fails
            text=True,   # Use text mode instead of bytes
            capture_output=False  # Stream output directly to console
        )

        print(f"\n✓ {description} - COMPLETE!")
        return True

    except subprocess.CalledProcessError as e:
        # Script returned non-zero exit code
        print(f"\n✗ Error executing {description}")
        print(f"Exit code: {e.returncode}")
        return False
    except Exception as e:
        # Unexpected error (file not found, permission denied, etc.)
        print(f"\n✗ Unexpected error: {str(e)}")
        return False


def main():
    """Main function orchestrating the complete dataset preparation process.

    This function coordinates the multi-step dataset preparation pipeline:
    1. Extracts hyperlinks and processes category keywords from Excel
    2. Downloads image files from Google Drive in parallel
    3. Generates final dataset with local file paths

    Returns:
        bool: True if all steps succeed, False if any step fails.
    """
    print("\n" + "="*80)
    print("PREPARAÇÃO DO DATASET - BLOCO SUVACO DE CRISTO")
    print("="*80)

    # Determine project root and script locations
    project_root = Path(__file__).parent.parent.parent
    dataset_dir = project_root / "src" / "core" / "dataset"

    # Define paths to processing scripts
    process_script = dataset_dir / "process.py"
    download_script = dataset_dir / "download.py"

    # Validate that required scripts exist
    if not process_script.exists():
        print(f"✗ Erro: Script não encontrado: {process_script}")
        return False

    if not download_script.exists():
        print(f"✗ Erro: Script não encontrado: {download_script}")
        return False

    # Step 1: Process data (extract hyperlinks and process categories)
    print("\nPASSO 1: Processamento de dados (hyperlinks e categorias)")
    print("-" * 80)
    success = run_script(
        str(process_script),
        "Processamento de dados"
    )

    if not success:
        print("\n✗ Falha no processamento de dados. Abortando...")
        return False

    # Step 2: Download files from Google Drive
    print("\n\nPASSO 2: Download de arquivos do Google Drive")
    print("-" * 80)
    success = run_script(
        str(download_script),
        "Download de arquivos"
    )

    if not success:
        print("\n✗ Falha no download de arquivos. Verifique os logs acima.")
        return False

    # Step 3: Re-run processing to generate final dataset with downloaded files
    print("\n\nPASSO 3: Gerando dataset final com arquivos baixados")
    print("-" * 80)
    success = run_script(
        str(process_script),
        "Geração do dataset final"
    )

    if not success:
        print("\n✗ Falha na geração do dataset final. Verifique os logs acima.")
        return False

    # Finalização
    print("\n" + "="*80)
    print("✓ PREPARAÇÃO DO DATASET CONCLUÍDA COM SUCESSO!")
    print("="*80)
    print("\nArquivos gerados:")
    print("  - dataset/catalogacao_with_hyperlinks.csv")
    print("  - dataset/palavras_chave.csv")
    print("  - dataset/palavras_chave_processed.csv")
    print("  - dataset/catalogacao_with_local_paths.csv")
    print("  - dataset/dataset_completo.csv")
    print("  - dataset/downloaded/ (arquivos baixados)")
    print("\n✓ Pronto para análise!")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
