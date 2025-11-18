"""Multilabel classification models with Bayesian optimization.

This module provides a unified interface for training and evaluating
multiple types of multilabel classifiers with hyperparameter optimization using Optuna.
"""
import pickle
from pathlib import Path
import warnings
import optuna
from optuna.samplers import TPESampler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import numpy as np
try:
    from skmultilearn.model_selection import IterativeStratification
except ImportError as exc:
    raise ImportError(
        "scikit-multilearn is required for iterative stratified K-fold. "
        "Install it with `pip install scikit-multilearn`."
    ) from exc

from ..evaluation.classification_metrics import calculate_multilabel_metrics


class MultilabelClassifierFactory:
    """Factory for creating multilabel classifiers with hyperparameter spaces."""

    AVAILABLE_CLASSIFIERS = {
        'lda': {
            'name': 'Linear Discriminant Analysis',
            'class': LinearDiscriminantAnalysis,
            'param_space': {
                'solver': ('categorical', ['svd', 'lsqr', 'eigen']),
                'shrinkage': ('categorical', [None, 'auto'])
            },
            'fixed_params': {}
        },
        'rf': {
            'name': 'Random Forest',
            'class': RandomForestClassifier,
            'param_space': {
                'n_estimators': ('int', 50, 300),
                'max_depth': ('int', 5, 30),
                'min_samples_split': ('int', 2, 20),
                'min_samples_leaf': ('int', 1, 10),
                'max_features': ('categorical', ['sqrt', 'log2'])
            },
            'fixed_params': {'random_state': 42, 'n_jobs': -1}
        },
        'xgb': {
            'name': 'XGBoost',
            'class': xgb.XGBClassifier,
            'param_space': {
                'n_estimators': ('int', 50, 300),
                'max_depth': ('int', 3, 10),
                'learning_rate': ('loguniform', 0.01, 0.3),
                'subsample': ('uniform', 0.6, 1.0),
                'colsample_bytree': ('uniform', 0.6, 1.0)
            },
            'fixed_params': {'random_state': 42, 'tree_method': 'hist', 'objective': 'binary:logistic', 'base_score': 0.5}
        },
        'lgbm': {
            'name': 'LightGBM',
            'class': lgb.LGBMClassifier,
            'param_space': {
                'n_estimators': ('int', 50, 300),
                'max_depth': ('int', 3, 10),
                'learning_rate': ('loguniform', 0.01, 0.3),
                'num_leaves': ('int', 20, 100),
                'subsample': ('uniform', 0.6, 1.0)
            },
            'fixed_params': {'random_state': 42, 'verbose': -1}
        },
        'nb': {
            'name': 'Naive Bayes',
            'class': GaussianNB,
            'param_space': {
                'var_smoothing': ('loguniform', 1e-10, 1e-5)
            },
            'fixed_params': {}
        },
        'knn': {
            'name': 'K-Nearest Neighbors',
            'class': KNeighborsClassifier,
            'param_space': {
                'n_neighbors': ('int', 3, 100),
                'weights': ('categorical', ['uniform', 'distance']),
                'metric': ('categorical', ['euclidean', 'manhattan', 'cosine'])
            },
            'fixed_params': {'n_jobs': -1}
        },
        'tree': {
            'name': 'Decision Tree',
            'class': DecisionTreeClassifier,
            'param_space': {
                'max_depth': ('int', 5, 50),
                'min_samples_split': ('int', 2, 20),
                'min_samples_leaf': ('int', 1, 10),
                'criterion': ('categorical', ['gini', 'entropy'])
            },
            'fixed_params': {'random_state': 42}
        },
        'svm': {
            'name': 'Linear SVM',
            'class': LinearSVC,
            'param_space': {
                'C': ('loguniform', 0.001, 100),
                'loss': ('categorical', ['hinge', 'squared_hinge']),
                'dual': ('categorical', [True, False])
            },
            'fixed_params': {'random_state': 42, 'max_iter': 10000}
        },
        'logistic': {
            'name': 'Logistic Regression',
            'class': LogisticRegression,
            'param_space': {
                'C': ('loguniform', 0.001, 100),
                'penalty': ('categorical', ['l2', None]),
                'solver': ('categorical', ['lbfgs', 'sag' ,'saga'])
            },
            'fixed_params': {'random_state': 42, 'max_iter': 10000, 'n_jobs': -1}
        },
        'passive_aggressive': {
            'name': 'Passive Aggressive Classifier',
            'class': PassiveAggressiveClassifier,
            'param_space': {
                'C': ('loguniform', 0.001, 100),
                'loss': ('categorical', ['hinge', 'squared_hinge'])
            },
            'fixed_params': {'random_state': 42, 'n_jobs': -1}
        }
    }

    @classmethod
    def create(cls, classifier_type, **kwargs):
        """Create a classifier by type.

        Args:
            classifier_type (str): Type of classifier ('logistic', 'knn', 'tree').
            **kwargs: Additional parameters for the classifier.

        Returns:
            MultiOutputClassifier: Configured multilabel classifier.
        """
        if classifier_type not in cls.AVAILABLE_CLASSIFIERS:
            raise ValueError(
                f"Unknown classifier: {classifier_type}. "
                f"Available: {list(cls.AVAILABLE_CLASSIFIERS.keys())}"
            )

        clf_info = cls.AVAILABLE_CLASSIFIERS[classifier_type]
        params = clf_info['fixed_params'].copy()
        params.update(kwargs)

        base_classifier = clf_info['class'](**params)
        return MultiOutputClassifier(base_classifier, n_jobs=-1)

    @classmethod
    def suggest_hyperparameters(cls, trial, classifier_type):
        """Suggest hyperparameters for Optuna trial.

        Args:
            trial (optuna.Trial): Optuna trial object.
            classifier_type (str): Type of classifier.

        Returns:
            dict: Dictionary of suggested hyperparameters.
        """
        clf_info = cls.AVAILABLE_CLASSIFIERS[classifier_type]
        params = {}

        for param_name, param_config in clf_info['param_space'].items():
            param_type = param_config[0]

            if param_type == 'loguniform':
                params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2], log=True)
            elif param_type == 'uniform':
                params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2])
            elif param_type == 'int':
                params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_config[1])

        # Handle special cases for LDA
        if classifier_type == 'lda':
            # shrinkage is only compatible with lsqr and eigen solvers
            if params['solver'] == 'svd':
                params['shrinkage'] = None

        # Handle special cases for LogisticRegression
        if classifier_type == 'logistic':
            # l1 penalty only works with liblinear and saga solvers
            if params['penalty'] == 'l1':
                if params['solver'] == 'lbfgs':
                    params['solver'] = 'saga'

        return params

    @classmethod
    def get_available_classifiers(cls):
        """Get list of available classifier types."""
        return list(cls.AVAILABLE_CLASSIFIERS.keys())

    @classmethod
    def get_classifier_info(cls, classifier_type):
        """Get information about a classifier type."""
        return cls.AVAILABLE_CLASSIFIERS.get(classifier_type)


class MultilabelClassifierPipeline:
    """Pipeline for training and evaluating multilabel classifiers with Bayesian optimization.

    Attributes:
        classifier: The trained classifier.
        classifier_type (str): Type of classifier.
        categories (list): List of category names.
        best_params (dict): Best hyperparameters found during optimization.
        optimization_history (list): History of optimization trials.
    """

    def __init__(self, classifier_type='logistic', categories=None):
        """Initialize classifier pipeline.

        Args:
            classifier_type (str): Type of classifier.
            categories (list, optional): List of category names.
        """
        self.classifier_type = classifier_type
        self.classifier = None
        self.categories = categories or []
        self.best_params = None
        self.optimization_history = []
        self.is_trained = False

    def optimize_and_train(self, X_train, y_train, n_trials=50, cv=3, optimize_for='f1_macro', verbose=True):
        """Optimize hyperparameters using Bayesian optimization and train final model.

        Args:
            X_train (np.ndarray): Training features (n_samples, n_features).
            y_train (np.ndarray): Training labels (n_samples, n_labels).
            n_trials (int): Number of optimization trials.
            cv (int): Number of cross-validation folds.
            optimize_for (str): Metric to optimize ('f1_micro', 'f1_macro', 'accuracy_micro').
            verbose (bool): Print optimization progress.

        Returns:
            self: Trained pipeline with optimized hyperparameters.
        """
        if verbose:
            print(f"\nOptimizing {MultilabelClassifierFactory.get_classifier_info(self.classifier_type)['name']}...")
            print(f"  Trials: {n_trials}")
            print(f"  CV folds: {cv}")
            print(f"  Metric: {optimize_for}")

        def objective(trial):
            """Objective function for Optuna."""
            # Suggest hyperparameters
            params = MultilabelClassifierFactory.suggest_hyperparameters(trial, self.classifier_type)

            X_arr = np.asarray(X_train)
            y_arr = np.asarray(y_train)

            # Deterministic shuffling before stratified split
            rng = np.random.RandomState(42)
            permutation = rng.permutation(len(X_arr))
            X_perm = X_arr[permutation]
            y_perm = y_arr[permutation]

            splitter = IterativeStratification(n_splits=cv, order=1)

            fold_scores = []

            for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_perm, y_perm), start=1):
                clf = MultilabelClassifierFactory.create(self.classifier_type, **params)

                x_tr = X_perm[train_idx]
                y_tr = y_perm[train_idx]
                x_val = X_perm[val_idx]
                y_val = y_perm[val_idx]

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Only one sample available")
                    warnings.filterwarnings("ignore", message="invalid value encountered")

                    try:
                        clf.fit(x_tr, y_tr)
                        y_pred = clf.predict(x_val)
                        y_pred = np.asarray(y_pred)
                    except Exception as exc:
                        if verbose:
                            print(f"  ⚠ Fold {fold_idx} failed during training/prediction ({exc}). Using score 0.0")
                        fold_scores.append(0.0)
                        continue

                try:
                    if optimize_for == 'f1_micro':
                        fold_score = f1_score(y_val, y_pred, average='micro', zero_division=0)
                    elif optimize_for == 'f1_macro':
                        fold_score = f1_score(y_val, y_pred, average='macro', zero_division=0)
                    elif optimize_for == 'accuracy_micro':
                        fold_score = accuracy_score(y_val, y_pred)
                    else:
                        raise ValueError(f"Unsupported optimize_for metric: {optimize_for}")
                except Exception as exc:
                    if verbose:
                        print(f"  ⚠ Fold {fold_idx} metric computation failed ({exc}). Using score 0.0")
                    fold_score = 0.0

                if np.isnan(fold_score):
                    if verbose:
                        print(f"  ⚠ Fold {fold_idx} metric returned NaN. Using score 0.0")
                    fold_score = 0.0

                fold_scores.append(float(fold_score))

            valid_scores = [s for s in fold_scores if not np.isnan(s)]

            if not valid_scores:
                if verbose:
                    print("  ⚠ All folds produced invalid scores. Returning 0.0")
                return 0.0

            return float(np.mean(valid_scores))

        # Create Optuna study
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler
        )

        # Suppress Optuna logs if not verbose
        optuna.logging.set_verbosity(optuna.logging.WARNING if not verbose else optuna.logging.INFO)

        # Run optimization
        study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

        # Store best parameters
        self.best_params = study.best_params
        self.optimization_history = [
            {'trial': t.number, 'value': t.value, 'params': t.params}
            for t in study.trials
        ]

        if verbose:
            print(f"\n✓ Optimization complete!")
            print(f"  Best {optimize_for}: {study.best_value:.4f}")
            print(f"  Best parameters:")
            for key, value in self.best_params.items():
                print(f"    {key}: {value}")

        # Train final model with best parameters
        if verbose:
            print(f"\nTraining final model with best parameters...")

        self.classifier = MultilabelClassifierFactory.create(self.classifier_type, **self.best_params)
        self.classifier.fit(X_train, y_train)
        self.is_trained = True

        if verbose:
            print("✓ Training complete!")

        return self

    def train(self, X_train, y_train, params=None):
        """Train the classifier with given or default parameters (without optimization).

        Args:
            X_train (np.ndarray): Training features (n_samples, n_features).
            y_train (np.ndarray): Training labels (n_samples, n_labels).
            params (dict, optional): Hyperparameters for the classifier.

        Returns:
            self: Trained pipeline.
        """
        print(f"Training {MultilabelClassifierFactory.get_classifier_info(self.classifier_type)['name']}...")

        params = params or {}
        self.best_params = params
        self.classifier = MultilabelClassifierFactory.create(self.classifier_type, **params)
        self.classifier.fit(X_train, y_train)
        self.is_trained = True

        return self

    def predict(self, X):
        """Predict labels for samples.

        Args:
            X (np.ndarray): Features (n_samples, n_features).

        Returns:
            np.ndarray: Predicted labels (n_samples, n_labels).
        """
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before prediction")
        return self.classifier.predict(X)

    def evaluate(self, X_test, y_test, verbose=True):
        """Evaluate classifier on test set using classification_metrics module.

        Args:
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test labels.
            verbose (bool): Print metrics summary.

        Returns:
            dict: Dictionary of evaluation metrics.
        """
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before evaluation")

        # Get predictions
        y_pred = self.predict(X_test)

        # Calculate exact match accuracy (subset accuracy) using sklearn
        # Fraction of samples where ALL labels are predicted correctly
        from sklearn.metrics import accuracy_score
        exact_match_accuracy = accuracy_score(y_test, y_pred)

        # Convert arrays to dictionary format for metrics module
        y_true_dict = {cat: y_test[:, i].tolist() for i, cat in enumerate(self.categories)}
        y_pred_dict = {cat: y_pred[:, i].tolist() for i, cat in enumerate(self.categories)}

        # Calculate metrics using classification_metrics module
        metrics_dict = calculate_multilabel_metrics(y_true_dict, y_pred_dict)

        # Extract summary metrics
        summary = {
            'exact_match_accuracy': exact_match_accuracy,  # NEW: Traditional accuracy
            'accuracy_micro': metrics_dict['micro_avg']['accuracy'],
            'precision_micro': metrics_dict['micro_avg']['precision'],
            'recall_micro': metrics_dict['micro_avg']['recall'],
            'f1_micro': metrics_dict['micro_avg']['f1'],
            'precision_macro': metrics_dict['macro_avg']['precision'],
            'recall_macro': metrics_dict['macro_avg']['recall'],
            'f1_macro': metrics_dict['macro_avg']['f1'],
            'per_category': {}
        }

        # Add per-category metrics
        for cat in self.categories:
            if cat in metrics_dict:
                summary['per_category'][cat] = metrics_dict[cat]

        if verbose:
            self._print_metrics(summary)

        return summary

    def _print_metrics(self, metrics):
        """Print evaluation metrics in a formatted way."""
        print(f"\n{'='*60}")
        print(f"Metrics for {MultilabelClassifierFactory.get_classifier_info(self.classifier_type)['name']}")
        print(f"{'='*60}")

        print("\nExact Match (all labels correct):")
        print(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f}")

        print("\nMicro-averaged (overall performance):")
        print(f"  Accuracy:           {metrics['accuracy_micro']:.4f}")
        print(f"  Precision:          {metrics['precision_micro']:.4f}")
        print(f"  Recall:             {metrics['recall_micro']:.4f}")
        print(f"  F1 Score:           {metrics['f1_micro']:.4f}")

        print("\nMacro-averaged (per-class average):")
        print(f"  Precision:          {metrics['precision_macro']:.4f}")
        print(f"  Recall:             {metrics['recall_macro']:.4f}")
        print(f"  F1 Score:           {metrics['f1_macro']:.4f}")

        # Show top/bottom 5 categories by F1
        if self.categories and 'per_category' in metrics:
            sorted_cats = sorted(
                self.categories,
                key=lambda c: metrics['per_category'][c]['f1'],
                reverse=True
            )

            print("\nTop 5 categories by F1:")
            for cat in sorted_cats[:5]:
                m = metrics['per_category'][cat]
                print(f"  {cat:25s} F1: {m['f1']:.4f} (P: {m['precision']:.3f}, R: {m['recall']:.3f})")

            if len(self.categories) > 10:
                print("\nBottom 5 categories by F1:")
                for cat in sorted_cats[-5:]:
                    m = metrics['per_category'][cat]
                    print(f"  {cat:25s} F1: {m['f1']:.4f} (P: {m['precision']:.3f}, R: {m['recall']:.3f})")

    def save(self, output_path):
        """Save trained classifier to file."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained classifier")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'classifier': self.classifier,
            'classifier_type': self.classifier_type,
            'categories': self.categories,
            'best_params': self.best_params,
            'optimization_history': self.optimization_history
        }

        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"✓ Model saved: {output_path}")

    @classmethod
    def load(cls, model_path):
        """Load trained classifier from file."""
        with open(model_path, 'rb') as f:
            save_data = pickle.load(f)

        pipeline = cls(
            classifier_type=save_data['classifier_type'],
            categories=save_data['categories']
        )
        pipeline.classifier = save_data['classifier']
        pipeline.best_params = save_data.get('best_params', {})
        pipeline.optimization_history = save_data.get('optimization_history', [])
        pipeline.is_trained = True

        return pipeline

    def __repr__(self):
        status = "trained" if self.is_trained else "untrained"
        opt_status = f", optimized" if self.best_params else ""
        return f"MultilabelClassifierPipeline({self.classifier_type}, {status}{opt_status})"
