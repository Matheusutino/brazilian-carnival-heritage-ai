"""Evaluation module for image classification and upscaling quality assessment.

This module provides comprehensive evaluation capabilities for both zero-shot
and LLM-based image classification systems, as well as upscaling quality assessment.
"""
from .classification_metrics import calculate_binary_metrics, calculate_multilabel_metrics

try:
    from .evaluator import ClassificationEvaluator
    from .report import ReportGenerator
    has_classification = True
except ImportError:
    has_classification = False

from .upscale_metrics import ImageQualityEvaluator

__all__ = [
    'calculate_binary_metrics',
    'calculate_multilabel_metrics',
    'ImageQualityEvaluator'
]

if has_classification:
    __all__.extend(['ClassificationEvaluator', 'ReportGenerator'])
