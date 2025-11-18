"""Train supervised multilabel classifiers using extracted features.

This script orchestrates the complete training pipeline:
1. Load dataset split
2. Extract features using pre-trained models
3. Train multiple classifiers
4. Evaluate and save results
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.data.dataset_loader import MultilabelDataset
from src.core.features.feature_extractor import FeatureExtractionPipeline
from src.core.classification.multilabel_classifiers import (
    MultilabelClassifierPipeline,
    MultilabelClassifierFactory
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train supervised multilabel classifiers'
    )

    parser.add_argument('--split', default='dataset/splits/split_latest.json',
                       help='Train/test split JSON file')
    parser.add_argument('--feature-model',
                       choices=['dinov2-small', 'dinov2-base',
                               'convnextv2-tiny', 'convnextv2-base'],
                       default='dinov2-small',
                       help='Feature extraction model')
    parser.add_argument('--classifiers', nargs='+',
                       choices=MultilabelClassifierFactory.get_available_classifiers(),
                       default=['lda', 'rf', 'xgb', 'lgbm', 'nb'],
                       help='Classifiers to train (available: lda, rf, tree, xgb, lgbm, nb, knn, svm, logistic, passive_aggressive)')
    parser.add_argument('--output-dir', default='results/classification/supervised',
                       help='Output directory for trained models')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for feature extraction')
    parser.add_argument('--device', default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--optimize', action='store_true',
                       help='Use Bayesian optimization to find best hyperparameters')
    parser.add_argument('--n-trials', type=int, default=20,
                       help='Number of Optuna trials for optimization (default: 20)')
    parser.add_argument('--cv-folds', type=int, default=3,
                       help='Number of cross-validation folds for optimization (default: 3)')

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    print("="*80)
    print("SUPERVISED MULTILABEL CLASSIFICATION PIPELINE")
    print("="*80)
    print(f"Feature model: {args.feature_model}")
    print(f"Classifiers: {', '.join(args.classifiers)}")
    print(f"Split file: {args.split}")

    # Step 1: Load dataset
    print("\n" + "="*80)
    print("STEP 1: Loading Dataset")
    print("="*80)

    dataset = MultilabelDataset(args.split, labels_source=None)
    print(f"\n{dataset}")

    train_paths, y_train = dataset.get_train_data()
    test_paths, y_test = dataset.get_test_data()
    categories = dataset.get_categories()

    print(f"\nTrain samples: {len(train_paths)}")
    print(f"Test samples: {len(test_paths)}")
    print(f"Categories: {len(categories)}")

    # Print label distribution
    print("\nLabel distribution:")
    dist = dataset.get_label_distribution()
    for cat in categories[:5]:  # Show first 5
        train_count = dist['train'][cat]
        test_count = dist['test'][cat]
        print(f"  {cat:20s} Train: {train_count:4d} | Test: {test_count:4d}")
    if len(categories) > 5:
        print(f"  ... and {len(categories) - 5} more categories")

    # Step 2: Extract features
    print("\n" + "="*80)
    print("STEP 2: Feature Extraction")
    print("="*80)

    print(f"\nInitializing feature extractor: {args.feature_model}")
    feature_pipeline = FeatureExtractionPipeline(args.feature_model, device=args.device)
    print(f"Feature dimension: {feature_pipeline.feature_dim}")

    print(f"\nExtracting train features...")
    X_train = feature_pipeline.extract_from_paths(train_paths, args.batch_size)

    print(f"\nExtracting test features...")
    X_test = feature_pipeline.extract_from_paths(test_paths, args.batch_size)

    print(f"\nTrain features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")

    # Step 3: Train and evaluate classifiers
    print("\n" + "="*80)
    print("STEP 3: Training Classifiers")
    print("="*80)

    results = {}

    for clf_type in args.classifiers:
        print(f"\n{'='*80}")
        clf_name = MultilabelClassifierFactory.get_classifier_info(clf_type)['name']
        print(f"Classifier: {clf_name}")
        print(f"{'='*80}")

        # Initialize pipeline
        clf_pipeline = MultilabelClassifierPipeline(clf_type, categories=categories)

        # Train (with or without optimization)
        if args.optimize:
            clf_pipeline.optimize_and_train(
                X_train, y_train,
                n_trials=args.n_trials,
                cv=args.cv_folds,
                optimize_for='f1_macro',
                verbose=True
            )
        else:
            clf_pipeline.train(X_train, y_train)

        # Get predictions
        print("\nGenerating predictions on test set...")
        y_pred = clf_pipeline.predict(X_test)

        # Create classifier-specific output directory
        clf_output_dir = Path(args.output_dir) / args.feature_model / clf_type
        clf_output_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions to results/ directory
        import numpy as np
        predictions_file = clf_output_dir / "test_predictions.npz"
        np.savez(predictions_file,
                 y_true=y_test,
                 y_pred=y_pred,
                 test_paths=test_paths,
                 categories=categories)
        print(f"✓ Predictions saved: {predictions_file}")

        # Save optimization history if available
        if clf_pipeline.optimization_history:
            optuna_file = clf_output_dir / "optuna_history.json"
            optuna_data = {
                'best_params': clf_pipeline.best_params,
                'n_trials': len(clf_pipeline.optimization_history),
                'trials': clf_pipeline.optimization_history
            }
            optuna_file.write_text(json.dumps(optuna_data, indent=2), encoding='utf-8')
            print(f"✓ Optuna history saved: {optuna_file}")

        # Save classifier-specific summary
        clf_summary_file = clf_output_dir / "summary.json"
        clf_summary = {
            'timestamp': datetime.now().isoformat(),
            'feature_model': args.feature_model,
            'classifier_type': clf_type,
            'classifier_name': clf_name,
            'best_params': clf_pipeline.best_params,
            'optimization_trials': len(clf_pipeline.optimization_history) if clf_pipeline.optimization_history else 0,
            'n_train': len(train_paths),
            'n_test': len(test_paths),
            'n_categories': len(categories),
            'categories': categories
        }
        clf_summary_file.write_text(json.dumps(clf_summary, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"✓ Summary saved: {clf_summary_file}")

        # Store results for global summary
        results[clf_type] = {
            'best_params': clf_pipeline.best_params,
            'optimization_trials': len(clf_pipeline.optimization_history) if clf_pipeline.optimization_history else 0
        }

    # Step 4: Save results summary
    print("\n" + "="*80)
    print("STEP 4: Saving Results Summary")
    print("="*80)

    # Save summary to results/ directory
    results_output_dir = Path(args.output_dir) / args.feature_model
    results_output_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'feature_model': args.feature_model,
        'feature_dim': X_train.shape[1],
        'n_train': len(train_paths),
        'n_test': len(test_paths),
        'n_categories': len(categories),
        'categories': categories,
        'classifiers': results
    }

    results_file.write_text(json.dumps(results_summary, indent=2), encoding='utf-8')
    print(f"\n✓ Summary saved: {results_file}")

    # Also save as latest
    latest_file = results_output_dir / "summary_latest.json"
    latest_file.write_text(json.dumps(results_summary, indent=2), encoding='utf-8')
    print(f"✓ Latest summary: {latest_file}")

    print("\n" + "="*80)
    print("✓ Training pipeline completed!")
    print("="*80)
    print("\nPredictions saved for each classifier.")
    print("Run evaluation script to compute metrics.")


if __name__ == "__main__":
    main()
