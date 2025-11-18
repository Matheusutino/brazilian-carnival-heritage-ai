"""Dataset loading and management for multilabel classification.

This module handles loading train/test splits and associated data.
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np


class MultilabelDataset:
    """Container for multilabel classification dataset.

    Attributes:
        train_paths (list): List of training image paths.
        test_paths (list): List of test image paths.
        y_train (np.ndarray): Training labels (n_train, n_labels).
        y_test (np.ndarray): Test labels (n_test, n_labels).
        categories (list): List of category names.
        split_info (dict): Split metadata.
    """

    def __init__(self, split_file, labels_source=None):
        """Initialize dataset from split file.

        Args:
            split_file (str or Path): Path to split JSON file.
            labels_source (str or dict, optional): Source of labels. Can be:
                - Path to labels file
                - Dictionary mapping image paths to labels
                - None to load from split_info (default and recommended)
        """
        self.split_file = Path(split_file)

        # Load split information
        with open(self.split_file, 'r') as f:
            self.split_info = json.load(f)

        self.categories = self.split_info['categories_list']
        train_indices = self.split_info['train_indices']
        test_indices = self.split_info['test_indices']

        # Load main dataset
        dataset_path = self.split_info['dataset']
        df = pd.read_csv(dataset_path)
        df = df[df['LocalPath'].notna()]

        # Filter only image files (MUST match create_train_test_split.py exactly!)
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
        df = df[df['LocalPath'].str.lower().str.endswith(tuple(image_extensions))]

        # Load labels from categories CSV (same way as create_train_test_split.py)
        if labels_source is None and 'categories' in self.split_info:
            # Load labels from the categories CSV used in split creation
            categories_path = self.split_info['categories']
            df_labels = pd.read_csv(categories_path)

            # Merge with main dataset on Item name
            df = df.merge(df_labels, left_on='Item', right_on='Item', how='inner', suffixes=('', '_label'))

            print(f"✓ Loaded real labels from: {categories_path}")

        # Get train/test splits
        df_train = df.iloc[train_indices]
        df_test = df.iloc[test_indices]

        self.train_paths = df_train['LocalPath'].tolist()
        self.test_paths = df_test['LocalPath'].tolist()

        # Extract label matrices
        self.y_train = df_train[self.categories].fillna(0).values.astype(int)
        self.y_test = df_test[self.categories].fillna(0).values.astype(int)

    def get_train_data(self):
        """Get training data.

        Returns:
            tuple: (train_paths, y_train)
        """
        return self.train_paths, self.y_train

    def get_test_data(self):
        """Get test data.

        Returns:
            tuple: (test_paths, y_test)
        """
        return self.test_paths, self.y_test

    def get_categories(self):
        """Get category names.

        Returns:
            list: List of category names.
        """
        return self.categories

    def get_n_samples(self):
        """Get number of samples in train and test sets.

        Returns:
            dict: Dictionary with 'train' and 'test' counts.
        """
        return {
            'train': len(self.train_paths),
            'test': len(self.test_paths)
        }

    def get_label_distribution(self):
        """Get label distribution statistics.

        Returns:
            dict: Dictionary with train and test label statistics.
        """
        train_dist = {
            cat: int(self.y_train[:, i].sum())
            for i, cat in enumerate(self.categories)
        }

        test_dist = {
            cat: int(self.y_test[:, i].sum())
            for i, cat in enumerate(self.categories)
        }

        return {
            'train': train_dist,
            'test': test_dist
        }

    def __repr__(self):
        return (f"MultilabelDataset(train={len(self.train_paths)}, "
                f"test={len(self.test_paths)}, "
                f"categories={len(self.categories)})")
