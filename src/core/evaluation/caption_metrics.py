"""Caption similarity metrics calculation module.

This module provides functions to calculate various text similarity metrics
for comparing generated captions against reference descriptions, including:
- BLEU (1-4 grams)
- METEOR
- ROUGE-L
- BERTScore (using Portuguese BERT model)
"""
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

# Text similarity metrics
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score


def download_nltk_data():
    """Download required NLTK data if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK wordnet...")
        nltk.download('wordnet')

    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        print("Downloading NLTK omw-1.4...")
        nltk.download('omw-1.4')


def calculate_bleu_scores(reference: str, hypothesis: str) -> Dict[str, float]:
    """Calculate BLEU scores (1-4 grams) for a single pair.

    Args:
        reference (str): Reference (human) text.
        hypothesis (str): Hypothesis (generated) text.

    Returns:
        Dict[str, float]: Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores.

    Example:
        >>> ref = "a cat sitting on a mat"
        >>> hyp = "a cat on the mat"
        >>> scores = calculate_bleu_scores(ref, hyp)
        >>> print(f"BLEU-1: {scores['bleu_1']:.3f}")
    """
    # Tokenize
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())

    # Smoothing for short sentences
    smoothing = SmoothingFunction().method1

    bleu_scores = {}
    for n in range(1, 5):
        weights = tuple([1.0/n] * n + [0.0] * (4-n))
        bleu_scores[f'bleu_{n}'] = sentence_bleu(
            [ref_tokens],
            hyp_tokens,
            weights=weights,
            smoothing_function=smoothing
        )

    return bleu_scores


def calculate_meteor(reference: str, hypothesis: str) -> float:
    """Calculate METEOR score for a single pair.

    METEOR considers synonyms, stemming, and paraphrasing, providing
    better correlation with human judgments than BLEU.

    Args:
        reference (str): Reference (human) text.
        hypothesis (str): Hypothesis (generated) text.

    Returns:
        float: METEOR score (0-1).

    Example:
        >>> ref = "a cat sitting on a mat"
        >>> hyp = "a cat on the mat"
        >>> score = calculate_meteor(ref, hyp)
        >>> print(f"METEOR: {score:.3f}")
    """
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())

    return meteor_score([ref_tokens], hyp_tokens)


def calculate_rouge_l(reference: str, hypothesis: str, scorer=None) -> float:
    """Calculate ROUGE-L score for a single pair.

    ROUGE-L measures the longest common subsequence between texts.

    Args:
        reference (str): Reference (human) text.
        hypothesis (str): Hypothesis (generated) text.
        scorer (rouge_scorer.RougeScorer, optional): Reusable scorer object.
            If None, creates a new one.

    Returns:
        float: ROUGE-L F1 score (0-1).

    Example:
        >>> ref = "a cat sitting on a mat"
        >>> hyp = "a cat on the mat"
        >>> score = calculate_rouge_l(ref, hyp)
        >>> print(f"ROUGE-L: {score:.3f}")
    """
    if scorer is None:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    scores = scorer.score(reference, hypothesis)
    return scores['rougeL'].fmeasure


def calculate_bertscore_batch(references: List[str], hypotheses: List[str],
                              model: str = 'neuralmind/bert-base-portuguese-cased',
                              verbose: bool = False) -> Tuple[List[float], List[float], List[float]]:
    """Calculate BERTScore for a batch of pairs.

    BERTScore leverages pre-trained BERT embeddings to evaluate semantic
    similarity between texts, providing context-aware evaluation.

    Args:
        references (List[str]): List of reference texts.
        hypotheses (List[str]): List of hypothesis texts.
        model (str, optional): BERT model to use. Defaults to Portuguese BERT.
        verbose (bool, optional): Show progress. Defaults to False.

    Returns:
        Tuple[List[float], List[float], List[float]]:
            Tuple of (precision, recall, f1) score lists.

    Example:
        >>> refs = ["a cat sitting on a mat", "a dog in the park"]
        >>> hyps = ["a cat on the mat", "a dog at the park"]
        >>> P, R, F1 = calculate_bertscore_batch(refs, hyps)
        >>> print(f"Average BERTScore F1: {np.mean(F1):.3f}")
    """
    try:
        P, R, F1 = bert_score(
            hypotheses,
            references,
            lang='pt',
            model_type=model,
            verbose=verbose
        )
    except KeyError:
        print(f"Warning: BERTScore model '{model}' not recognized. Falling back to default model for Portuguese.")
        P, R, F1 = bert_score(
            hypotheses,
            references,
            lang='pt',
            model_type=None,
            verbose=verbose
        )
    return P.tolist(), R.tolist(), F1.tolist()


def evaluate_caption_pairs(references: List[str],
                           hypotheses: List[str],
                           filenames: List[str] = None,
                           bert_model: str = 'neuralmind/bert-base-portuguese-cased',
                           verbose: bool = True) -> Dict:
    """Evaluate generated captions against reference descriptions.

    Calculates multiple text similarity metrics for each reference-hypothesis pair
    and aggregates results with statistical summaries.

    Args:
        references (List[str]): List of reference (human) texts.
        hypotheses (List[str]): List of hypothesis (generated) texts.
        filenames (List[str], optional): List of filenames for per-image tracking.
        bert_model (str, optional): BERT model for BERTScore.
        verbose (bool, optional): Print progress messages. Defaults to True.

    Returns:
        Dict: Dictionary containing:
            - num_pairs (int): Number of evaluated pairs
            - metrics (Dict): Aggregated metrics with mean, std, min, max, median
            - per_image (List[Dict]): Per-image detailed metrics

    Example:
        >>> refs = ["a cat on a mat", "a dog in park"]
        >>> hyps = ["a cat sitting", "a dog playing"]
        >>> results = evaluate_caption_pairs(refs, hyps)
        >>> print(f"Average BLEU-4: {results['metrics']['bleu_4']['mean']:.3f}")
    """
    # Ensure NLTK data is available
    download_nltk_data()

    if len(references) != len(hypotheses):
        raise ValueError(f"References ({len(references)}) and hypotheses ({len(hypotheses)}) must have same length")

    if not references:
        if verbose:
            print("Warning: No pairs to evaluate")
        return {'num_pairs': 0, 'metrics': {}, 'per_image': []}

    if verbose:
        print(f"\nEvaluating {len(references)} caption pairs...")

    # Initialize rouge scorer (reuse for all pairs)
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    # Calculate metrics per pair
    all_metrics = defaultdict(list)

    for ref, hyp in zip(references, hypotheses):
        # BLEU scores
        bleu_scores = calculate_bleu_scores(ref, hyp)
        for key, value in bleu_scores.items():
            all_metrics[key].append(value)

        # METEOR
        meteor = calculate_meteor(ref, hyp)
        all_metrics['meteor'].append(meteor)

        # ROUGE-L
        rouge_l = calculate_rouge_l(ref, hyp, rouge_scorer_obj)
        all_metrics['rouge_l'].append(rouge_l)

    # Calculate BERTScore in batch
    if verbose:
        print("Calculating BERTScore (this may take a while)...")
    bert_p, bert_r, bert_f1 = calculate_bertscore_batch(references, hypotheses, model=bert_model, verbose=verbose)
    all_metrics['bertscore_precision'] = bert_p
    all_metrics['bertscore_recall'] = bert_r
    all_metrics['bertscore_f1'] = bert_f1

    # Aggregate metrics
    results = {
        'num_pairs': len(references),
        'metrics': {}
    }

    for metric_name, values in all_metrics.items():
        results['metrics'][metric_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values))
        }

    # Store per-image results
    if filenames is None:
        filenames = [f"image_{i}" for i in range(len(references))]

    results['per_image'] = [
        {
            'filename': filename,
            'reference': ref,
            'hypothesis': hyp,
            'bleu_1': all_metrics['bleu_1'][i],
            'bleu_2': all_metrics['bleu_2'][i],
            'bleu_3': all_metrics['bleu_3'][i],
            'bleu_4': all_metrics['bleu_4'][i],
            'meteor': all_metrics['meteor'][i],
            'rouge_l': all_metrics['rouge_l'][i],
            'bertscore_precision': all_metrics['bertscore_precision'][i],
            'bertscore_recall': all_metrics['bertscore_recall'][i],
            'bertscore_f1': all_metrics['bertscore_f1'][i]
        }
        for i, (filename, ref, hyp) in enumerate(zip(filenames, references, hypotheses))
    ]

    return results


def print_metrics_summary(results: Dict, caption_type: str = None):
    """Print evaluation metrics in a formatted table.

    Args:
        results (Dict): Results dictionary from evaluate_caption_pairs.
        caption_type (str, optional): Type label for display.
    """
    print(f"\n{'='*80}")
    if caption_type:
        print(f"Results for: {caption_type}")
    print(f"Number of pairs evaluated: {results['num_pairs']}")
    print(f"{'='*80}\n")

    print(f"{'Metric':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
    print(f"{'-'*80}")

    for metric_name, stats in results['metrics'].items():
        print(f"{metric_name:<25} "
              f"{stats['mean']:<10.4f} "
              f"{stats['std']:<10.4f} "
              f"{stats['min']:<10.4f} "
              f"{stats['max']:<10.4f} "
              f"{stats['median']:<10.4f}")
