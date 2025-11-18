"""Metrics calculation module for classification evaluation.

This module provides functions to calculate various classification metrics including
precision, recall, F1-score, accuracy, and confusion matrix components for both
binary and multilabel classification tasks.
"""
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    hamming_loss,
    jaccard_score
)


def calculate_binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Calculate binary classification metrics for a single category.

    Computes standard binary classification metrics including true positives,
    false positives, true negatives, false negatives, precision, recall,
    F1-score, and accuracy.

    Args:
        y_true (List[int]): Ground truth binary labels (0 or 1).
        y_pred (List[int]): Predicted binary labels (0 or 1).

    Returns:
        Dict[str, float]: Dictionary containing the following metrics:
            - tp (int): True positives
            - fp (int): False positives
            - tn (int): True negatives
            - fn (int): False negatives
            - precision (float): Precision score (0-1)
            - recall (float): Recall score (0-1)
            - f1 (float): F1-score (0-1)
            - accuracy (float): Accuracy score (0-1)
            - support (int): Number of positive samples in ground truth

    Example:
        >>> y_true = [1, 0, 1, 1, 0]
        >>> y_pred = [1, 0, 0, 1, 0]
        >>> metrics = calculate_binary_metrics(y_true, y_pred)
        >>> print(f"F1: {metrics['f1']:.3f}")
        F1: 0.800
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Calculate metrics using sklearn
    precision = precision_score(y_true, y_pred, zero_division=0.0)
    recall = recall_score(y_true, y_pred, zero_division=0.0)
    f1 = f1_score(y_true, y_pred, zero_division=0.0)
    accuracy = accuracy_score(y_true, y_pred)
    support = int(np.sum(y_true == 1))

    return {
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'support': support
    }


def calculate_multilabel_metrics(
    y_true_dict: Dict[str, List[int]],
    y_pred_dict: Dict[str, List[int]]
) -> Dict[str, Dict[str, float]]:
    """Calculate metrics for multilabel classification across all categories.

    Computes per-category metrics and aggregate metrics (micro, macro, weighted
    averages) for multilabel classification tasks.

    Args:
        y_true_dict (Dict[str, List[int]]): Dictionary mapping category names to
            ground truth binary labels.
        y_pred_dict (Dict[str, List[int]]): Dictionary mapping category names to
            predicted binary labels.

    Returns:
        Dict[str, Dict[str, float]]: Nested dictionary with two levels:
            - First level: category names plus 'micro_avg', 'macro_avg', 'weighted_avg'
            - Second level: metric names and their values

            Per-category metrics include: tp, fp, tn, fn, precision, recall, f1,
            accuracy, support

            Aggregate metrics include: precision, recall, f1, accuracy, total_support

    Example:
        >>> y_true = {
        ...     'cat1': [1, 0, 1],
        ...     'cat2': [0, 1, 1]
        ... }
        >>> y_pred = {
        ...     'cat1': [1, 0, 0],
        ...     'cat2': [0, 1, 1]
        ... }
        >>> metrics = calculate_multilabel_metrics(y_true, y_pred)
        >>> print(f"Macro F1: {metrics['macro_avg']['f1']:.3f}")
    """
    # Calculate per-category metrics
    category_metrics = {}

    for category in y_true_dict.keys():
        if category not in y_pred_dict:
            # Category missing in predictions, treat all as zeros
            y_pred = [0] * len(y_true_dict[category])
        else:
            y_pred = y_pred_dict[category]

        y_true = y_true_dict[category]
        category_metrics[category] = calculate_binary_metrics(y_true, y_pred)

    # Calculate aggregate metrics using sklearn
    all_categories = list(category_metrics.keys())

    # Convert to array format for sklearn functions
    n_samples = len(y_true_dict[all_categories[0]])
    y_true_array = np.array([[y_true_dict[cat][i] for cat in all_categories]
                              for i in range(n_samples)])
    y_pred_array = np.array([[y_pred_dict.get(cat, [0]*n_samples)[i] for cat in all_categories]
                              for i in range(n_samples)])

    total_support = sum(category_metrics[cat]['support'] for cat in all_categories)

    # Micro-average using sklearn
    category_metrics['micro_avg'] = {
        'precision': float(precision_score(y_true_array, y_pred_array, average='micro', zero_division=0.0)),
        'recall': float(recall_score(y_true_array, y_pred_array, average='micro', zero_division=0.0)),
        'f1': float(f1_score(y_true_array, y_pred_array, average='micro', zero_division=0.0)),
        'accuracy': float(accuracy_score(y_true_array, y_pred_array)),
        'total_support': total_support
    }

    # Macro-average using sklearn
    category_metrics['macro_avg'] = {
        'precision': float(precision_score(y_true_array, y_pred_array, average='macro', zero_division=0.0)),
        'recall': float(recall_score(y_true_array, y_pred_array, average='macro', zero_division=0.0)),
        'f1': float(f1_score(y_true_array, y_pred_array, average='macro', zero_division=0.0)),
        'accuracy': float(np.mean([category_metrics[cat]['accuracy'] for cat in all_categories])),
        'total_support': total_support
    }

    # Weighted-average using sklearn
    category_metrics['weighted_avg'] = {
        'precision': float(precision_score(y_true_array, y_pred_array, average='weighted', zero_division=0.0)),
        'recall': float(recall_score(y_true_array, y_pred_array, average='weighted', zero_division=0.0)),
        'f1': float(f1_score(y_true_array, y_pred_array, average='weighted', zero_division=0.0)),
        'accuracy': float(sum(
            category_metrics[cat]['accuracy'] * category_metrics[cat]['support']
            for cat in all_categories
        ) / total_support) if total_support > 0 else 0.0,
        'total_support': total_support
    }

    return category_metrics


def calculate_per_image_metrics(
    y_true_dict: Dict[str, List[int]],
    y_pred_dict: Dict[str, List[int]]
) -> Dict[str, float]:
    """Calculate per-image metrics for multilabel classification.

    Computes metrics treating each image as a multilabel classification instance,
    useful for understanding overall system performance across all categories
    simultaneously.

    Args:
        y_true_dict (Dict[str, List[int]]): Dictionary mapping category names to
            ground truth binary labels.
        y_pred_dict (Dict[str, List[int]]): Dictionary mapping category names to
            predicted binary labels.

    Returns:
        Dict[str, float]: Dictionary containing per-image metrics:
            - exact_match_ratio: Proportion of images with perfect predictions
            - hamming_loss: Average proportion of wrong labels per image
            - jaccard_score: Average Jaccard similarity (intersection over union)

    Example:
        >>> y_true = {'cat1': [1, 0], 'cat2': [0, 1]}
        >>> y_pred = {'cat1': [1, 0], 'cat2': [0, 1]}
        >>> metrics = calculate_per_image_metrics(y_true, y_pred)
        >>> print(f"Exact match: {metrics['exact_match_ratio']:.2%}")
        Exact match: 100.00%
    """
    categories = list(y_true_dict.keys())
    n_samples = len(y_true_dict[categories[0]])

    # Convert dict format to array format (n_samples, n_labels)
    y_true_array = np.array([[y_true_dict[cat][i] for cat in categories]
                              for i in range(n_samples)])
    y_pred_array = np.array([[y_pred_dict.get(cat, [0]*n_samples)[i] for cat in categories]
                              for i in range(n_samples)])

    # Calculate exact match ratio
    exact_matches = np.all(y_true_array == y_pred_array, axis=1).sum()
    exact_match_ratio = float(exact_matches / n_samples) if n_samples > 0 else 0.0

    # Calculate hamming loss using sklearn
    hamming = hamming_loss(y_true_array, y_pred_array)

    # Calculate jaccard score using sklearn (samples average)
    jaccard = jaccard_score(y_true_array, y_pred_array, average='samples', zero_division=0.0)

    return {
        'exact_match_ratio': exact_match_ratio,
        'hamming_loss': float(hamming),
        'jaccard_score': float(jaccard)
    }
