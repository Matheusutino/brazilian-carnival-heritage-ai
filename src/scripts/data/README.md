# 📊 Data Processing Scripts

This directory contains scripts for data preparation and dataset manipulation, primarily for creating train/test splits and data preprocessing.

## 📄 Scripts

### `create_train_test_split.py`

**Purpose**: Creates a stratified 70/30 train/test split for multilabel classification using iterative stratification.

**Key Features**:
- ✅ **Iterative Stratification**: Uses `skmultilearn.model_selection.iterative_train_test_split`
- ✅ **Label Distribution Preservation**: Maintains proportional distribution across all 23 categories
- ✅ **Reproducibility**: Fixed random state (42) for consistent splits
- ✅ **Metadata Export**: Saves split indices, statistics, and configuration
- ✅ **CSV Export**: Generates separate `train.csv` and `test.csv` files

**Algorithm**:
The script uses **Iterative Stratification** (Sechidis et al., 2011) specifically designed for multilabel datasets. This algorithm ensures that each subset has approximately the same percentage of samples for each label, handling label dependencies and co-occurrence patterns.

---

## 🚀 Usage

### Basic Usage

```bash
python -m src.scripts.data.create_train_test_split \
    --dataset dataset/dataset_completo.csv \
    --categories dataset/palavras_chave_processed.csv \
    --output dataset/splits/split_latest.json \
    --test-size 0.3 \
    --random-state 42
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--dataset` | `str` | `dataset/dataset_completo.csv` | Path to main dataset CSV |
| `--categories` | `str` | `dataset/palavras_chave_processed.csv` | Path to categories CSV with labels |
| `--output` | `str` | `dataset/splits/split_latest.json` | Output JSON file path |
| `--test-size` | `float` | `0.3` | Test set proportion (0.0-1.0) |
| `--random-state` | `int` | `42` | Random seed for reproducibility |

---

## 📥 Input Format

### Dataset CSV (`dataset_completo.csv`)

Expected columns:
- `Item`: Image identifier/name
- `LocalPath`: Path to the image file
- Other metadata columns

Example:
```csv
Item,LocalPath,Year,Description
img_001,dataset/images/img_001.jpg,2020,Carnival float
img_002,dataset/images/img_002.jpg,2021,Dancers
```

### Categories CSV (`palavras_chave_processed.csv`)

Expected structure:
- `Item`: Image identifier (matches dataset)
- `Id`: Unique label ID
- **23 category columns**: Binary labels (0/1) for each category

Categories:
```
adereços, baianas, bateria, comissão de frente, corpo coreográfico,
carro alegórico, carros abridores, carros principais, carros
intermediários, destaque de chão, destaque de carro, fantasia,
desfile, enredo, plateia, mestre sala e porta bandeira, passistas,
pessoas, público, porta bandeira, mestre sala, traje, véspera
```

Example:
```csv
Item,Id,adereços,baianas,bateria,comissão de frente,...
img_001,1,1,0,1,0,...
img_002,2,0,1,0,1,...
```

---

## 📤 Output Format

### Split JSON (`split_latest.json`)

Main output file containing all split information:

```json
{
  "timestamp": "2025-10-24T15:30:00",
  "dataset": "dataset/dataset_completo.csv",
  "categories": "dataset/palavras_chave_processed.csv",
  "n_total": 1000,
  "n_train": 700,
  "n_test": 300,
  "test_size": 0.3,
  "random_state": 42,
  "categories_list": ["adereços", "baianas", "bateria", ...],
  "train_indices": [0, 3, 5, 7, ...],
  "test_indices": [1, 2, 4, 6, ...]
}
```

### Train/Test CSV Files

The script also generates:
- `dataset/splits/train.csv`: Subset of images for training
- `dataset/splits/test.csv`: Subset of images for testing

Both files have the same structure as the input dataset CSV, with all columns preserved.

---

## 🔍 Processing Pipeline

The script follows these steps:

1. **Load Categories**: Extract category names from categories CSV
2. **Load Dataset**: Read main dataset CSV
3. **Filter Valid Images**:
   - Keep only rows with valid `LocalPath`
   - Keep only image files (`.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`, `.tiff`, `.tif`)
   - Exclude macOS metadata files (`._*`) and hidden files
4. **Load Labels**: Merge dataset with categories CSV on `Item` column
5. **Remove Duplicates**: Keep first occurrence of each `Item` and `LocalPath`
6. **Extract Label Matrix**: Create binary matrix `y` of shape `(n_images, 23)`
7. **Iterative Stratification**: Split using `iterative_train_test_split`
8. **Save Results**:
   - JSON with split indices and metadata
   - Separate train.csv and test.csv files

---

## 📊 Statistics Displayed

During execution, the script displays:

### Overall Statistics
```
Total images: 1000
Images with labels: 995
Images with at least one label: 990
Images with no labels: 5
Average labels per image: 3.24
```

### Label Distribution (Overall)
```
Label distribution:
  adereços: 450 (45.0%)
  baianas: 320 (32.0%)
  bateria: 280 (28.0%)
  ...
```

### Split Statistics
```
Train set: 700 images (70.0%)
Test set: 300 images (30.0%)
```

### Label Distribution (Train Set)
```
Label distribution in train set:
  adereços: 315 (45.0%)
  baianas: 224 (32.0%)
  bateria: 196 (28.0%)
  ...
```

### Label Distribution (Test Set)
```
Label distribution in test set:
  adereços: 135 (45.0%)
  baianas: 96 (32.0%)
  bateria: 84 (28.0%)
  ...
```

The goal is to maintain similar percentages across train and test sets for all labels.

---

## 🎯 Use Cases

### 1. Initial Dataset Split

Create the first train/test split for the project:

```bash
python -m src.scripts.data.create_train_test_split \
    --dataset dataset/dataset_completo.csv \
    --categories dataset/palavras_chave_processed.csv \
    --output dataset/splits/split_latest.json
```

### 2. Custom Test Size

Create a split with different proportions (e.g., 80/20):

```bash
python -m src.scripts.data.create_train_test_split \
    --test-size 0.2 \
    --output dataset/splits/split_80_20.json
```

### 3. Multiple Splits

Create multiple splits with different random seeds (for cross-validation or ensemble):

```bash
# Split 1
python -m src.scripts.data.create_train_test_split \
    --random-state 42 \
    --output dataset/splits/split_seed_42.json

# Split 2
python -m src.scripts.data.create_train_test_split \
    --random-state 123 \
    --output dataset/splits/split_seed_123.json

# Split 3
python -m src.scripts.data.create_train_test_split \
    --random-state 456 \
    --output dataset/splits/split_seed_456.json
```

---

## 🔧 Dependencies

```bash
pip install pandas numpy scikit-multilearn
```

**Libraries Used**:
- `pandas` — Data manipulation and CSV handling
- `numpy` — Numerical operations and array handling
- `scikit-multilearn` — Iterative stratification for multilabel data

**Note**: `scikit-multilearn` may require additional dependencies:
```bash
pip install scikit-learn scipy
```

---

## ⚠️ Important Notes

### Label Co-occurrence

Multilabel classification means images can have multiple labels simultaneously. Iterative stratification handles this by considering:
- Label frequency
- Label co-occurrence patterns
- Label dependencies

### Why Iterative Stratification?

Traditional stratification (e.g., `train_test_split` from scikit-learn) is designed for single-label classification. For multilabel problems, we need:

- **Proportional distribution**: Each label should have similar proportions in train and test
- **Co-occurrence preservation**: If labels A and B frequently appear together, this pattern should be maintained
- **Handling rare labels**: Ensure rare labels appear in both train and test sets

### Random State

The script defaults to `random_state=42` for consistency across experiments. Changing this value will produce different (but equally valid) splits.

### Duplicate Handling

The script removes duplicates based on:
1. `Item` column in categories CSV (keeps first occurrence)
2. `LocalPath` column in merged dataset (keeps first occurrence)

This ensures each image appears only once in either train or test set.

### Hidden Files and Metadata

The script automatically filters out:
- macOS metadata files (files starting with `._`)
- Hidden files (files starting with `.`)
- Non-image files

---

## 📈 Example Output

```
================================================================================
MULTILABEL TRAIN/TEST SPLIT
================================================================================
Dataset: dataset/dataset_completo.csv
Output: dataset/splits/split_latest.json
Test size: 0.3
Random state: 42

Categories: 23

Total images: 1000

Loading labels from categories CSV...
Images with labels: 995

Label matrix shape: (995, 23)
Images with at least one label: 990
Images with no labels: 5

Label matrix shape: (995, 23)
Average labels per image: 3.24
Label distribution:
  adereços: 450 (45.2%)
  baianas: 320 (32.2%)
  bateria: 280 (28.1%)
  ...

Performing iterative stratified split...

Train set: 697 images (70.1%)
Test set: 298 images (29.9%)

Label distribution in train set:
  adereços: 315 (45.2%)
  baianas: 224 (32.1%)
  bateria: 196 (28.1%)
  ...

Label distribution in test set:
  adereços: 135 (45.3%)
  baianas: 96 (32.2%)
  bateria: 84 (28.2%)
  ...

✓ Split saved: dataset/splits/split_latest.json
✓ Train CSV: dataset/splits/train.csv
✓ Test CSV: dataset/splits/test.csv

================================================================================
Split creation completed!
================================================================================
```

---

## 🔍 Troubleshooting

### Error: "No images found"

**Problem**: Dataset CSV is empty or all rows filtered out

**Solution**:
1. Check that `LocalPath` column contains valid paths
2. Verify image file extensions are valid
3. Ensure files are not hidden or metadata files

```bash
# Check dataset structure
head -n 5 dataset/dataset_completo.csv

# Count valid image paths
python -c "
import pandas as pd
df = pd.read_csv('dataset/dataset_completo.csv')
print(f'Total rows: {len(df)}')
print(f'Valid LocalPath: {df[\"LocalPath\"].notna().sum()}')
"
```

### Error: "Categories not found"

**Problem**: Categories CSV missing or malformed

**Solution**: Verify the categories CSV has the expected structure:

```bash
head -n 3 dataset/palavras_chave_processed.csv
```

Expected output:
```
Item,Id,adereços,baianas,bateria,...
img_001,1,1,0,1,...
img_002,2,0,1,0,...
```

### Error: "Not enough samples for split"

**Problem**: Dataset too small for the requested test size

**Solution**: Reduce `--test-size` or increase dataset size:

```bash
# Use smaller test size
python -m src.scripts.data.create_train_test_split \
    --test-size 0.2
```

### Warning: "Imbalanced label distribution"

**Problem**: Some labels are very rare (< 5 samples)

**Impact**: Iterative stratification may not be able to maintain perfect distribution for very rare labels

**Solution**:
- Accept slight imbalance for rare labels (usually acceptable)
- Consider data augmentation for rare categories
- Use larger dataset

---

## 📚 References

- **Iterative Stratification**: Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011). "On the stratification of multi-label data." *Machine Learning and Knowledge Discovery in Databases*, 145-158.
- **scikit-multilearn Documentation**: http://scikit.ml/stratification.html
- **Multilabel Classification**: Tsoumakas, G., & Katakis, I. (2007). "Multi-label classification: An overview." *International Journal of Data Warehousing and Mining*, 3(3), 1-13.

---

## 🤝 Contributing

To add new data processing scripts:

1. Follow the naming convention: `{action}_{object}.py` (e.g., `create_train_test_split.py`)
2. Include comprehensive docstrings and argument help text
3. Use argparse for command-line arguments with sensible defaults
4. Display informative statistics during execution
5. Export both machine-readable (JSON) and human-readable (CSV) outputs
6. Handle edge cases (empty datasets, missing files, invalid paths)
7. Update this README with the new script documentation

---

## 📧 Contact

For questions about data processing scripts, see the project's main documentation or open an issue in the repository.
