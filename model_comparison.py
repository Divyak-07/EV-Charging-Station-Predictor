# -*- coding: utf-8 -*-
"""
model_comparison.py - Multi-Model Comparison with Optuna Hyperparameter Tuning
===============================================================================
Compares Random Forest, XGBoost, and LightGBM classifiers for EV station
suitability prediction. Uses Optuna for automated hyperparameter optimization.

Usage:
    python model_comparison.py
    python model_comparison.py --trials 100
    python model_comparison.py --data data/global_training_data.csv

Outputs (saved to output/):
    10_model_comparison.png     - Grouped bar chart (AUC per model per fold)
    11_optuna_optimization.png  - Optimization history curves
    12_roc_curves.png           - Overlaid ROC curves for all 3 models
    model_comparison_report.json - Full results
    ev_model_best.joblib        - Best overall model
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, f1_score
)

try:
    import xgboost as xgb
except ImportError:
    sys.exit("ERROR: xgboost is required. Install: pip install xgboost")

try:
    import lightgbm as lgb
except ImportError:
    sys.exit("ERROR: lightgbm is required. Install: pip install lightgbm")

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    sys.exit("ERROR: optuna is required. Install: pip install optuna")

# Suppress verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
DEFAULT_DATA = PROJECT_ROOT / "output" / "training_data.csv"
GLOBAL_DATA = PROJECT_ROOT / "data" / "global_training_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Plot style
COLORS = {
    "Random Forest": "#6366f1",
    "XGBoost": "#10b981",
    "LightGBM": "#f59e0b",
}
BG_COLOR = "#0d1117"
CARD_COLOR = "#161b22"
TEXT_COLOR = "#e6edf3"
GRID_COLOR = "#21262d"


# ─────────────────────────────────────────────────────────────────────────────
# OPTUNA OBJECTIVE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def rf_objective(trial, X, y, cv):
    """Optuna objective for Random Forest."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    clf = RandomForestClassifier(**params)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    return scores.mean()


def xgb_objective(trial, X, y, cv):
    """Optuna objective for XGBoost."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 3.0),
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }
    clf = xgb.XGBClassifier(**params)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    return scores.mean()


def lgbm_objective(trial, X, y, cv):
    """Optuna objective for LightGBM."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", -1, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "is_unbalance": True,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    clf = lgb.LGBMClassifier(**params)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    return scores.mean()


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING & COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(data_path, n_trials=50, output_dir=OUTPUT_DIR):
    """Run full model comparison pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 64)
    print("  MULTI-MODEL COMPARISON WITH OPTUNA TUNING")
    print("=" * 64)

    # ── Load Data ──
    print(f"\n[1/6] Loading training data: {data_path}")
    df = pd.read_csv(data_path)
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    feature_names = [c for c in df.columns if c != "label"]
    print(f"   Samples: {len(X)} ({y.sum()} positive, {(y==0).sum()} negative)")
    print(f"   Features: {len(feature_names)}")

    # ── Cross-validation setup ──
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Optuna Optimization ──
    print(f"\n[2/6] Running Optuna optimization ({n_trials} trials per model)...")
    models_config = {
        "Random Forest": (rf_objective, RandomForestClassifier),
        "XGBoost": (xgb_objective, xgb.XGBClassifier),
        "LightGBM": (lgbm_objective, lgb.LGBMClassifier),
    }

    best_params = {}
    study_histories = {}

    for name, (objective_fn, _) in models_config.items():
        color = COLORS[name]
        print(f"\n   --- {name} ---")
        start = time.time()

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            study_name=name,
        )
        study.optimize(
            lambda trial: objective_fn(trial, X, y, cv),
            n_trials=n_trials,
            show_progress_bar=False,
        )

        elapsed = time.time() - start
        best_params[name] = study.best_params
        study_histories[name] = [t.value for t in study.trials if t.value is not None]
        print(f"   Best AUC: {study.best_value:.4f}  ({elapsed:.1f}s)")
        print(f"   Best params: {study.best_params}")

    # ── Train Final Models with Best Params ──
    print(f"\n[3/6] Training final models with best hyperparameters...")
    final_models = {}
    cv_results = {}

    for name in models_config:
        params = best_params[name].copy()
        params["random_state"] = 42
        params["n_jobs"] = -1

        if name == "Random Forest":
            params["class_weight"] = "balanced"
            clf = RandomForestClassifier(**params)
        elif name == "XGBoost":
            params["eval_metric"] = "auc"
            params["verbosity"] = 0
            clf = xgb.XGBClassifier(**params)
        else:  # LightGBM
            params["is_unbalance"] = True
            params["verbose"] = -1
            clf = lgb.LGBMClassifier(**params)

        # 5-fold CV scores
        scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        cv_results[name] = scores

        # Train on full data for final model
        clf.fit(X, y)
        final_models[name] = clf

        print(f"   {name:15s}  AUC: {scores.mean():.4f} +/- {scores.std():.4f}  "
              f"[{', '.join(f'{s:.3f}' for s in scores)}]")

    # ── Determine Best Model ──
    best_name = max(cv_results, key=lambda k: cv_results[k].mean())
    best_model = final_models[best_name]
    best_auc = cv_results[best_name].mean()

    print(f"\n   >> BEST MODEL: {best_name} (AUC = {best_auc:.4f})")

    # ── Save Best Model ──
    print(f"\n[4/6] Saving best model...")
    model_path = output_dir / "ev_model_best.joblib"
    joblib.dump(best_model, model_path)
    print(f"   Saved: {model_path}")

    # ── Generate Charts ──
    print(f"\n[5/6] Generating comparison charts...")
    plot_model_comparison(cv_results, output_dir)
    plot_optuna_history(study_histories, output_dir)
    plot_roc_curves(final_models, X, y, cv, output_dir)

    # ── Save JSON Report ──
    print(f"\n[6/6] Saving comparison report...")
    report = {
        "best_model": best_name,
        "best_auc": round(best_auc, 4),
        "training_samples": len(X),
        "n_features": len(feature_names),
        "n_optuna_trials": n_trials,
        "models": {},
    }
    for name in models_config:
        scores = cv_results[name]
        report["models"][name] = {
            "mean_auc": round(scores.mean(), 4),
            "std_auc": round(scores.std(), 4),
            "fold_scores": [round(s, 4) for s in scores],
            "best_params": best_params[name],
        }

    report_path = output_dir / "model_comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"   Saved: {report_path}")

    # ── Summary ──
    print(f"\n{'=' * 64}")
    print(f"  COMPARISON COMPLETE")
    print(f"{'=' * 64}")
    print(f"\n  Results:")
    print(f"  {'Model':<18} {'Mean AUC':>10} {'Std':>8}  {'Verdict':>10}")
    print(f"  {'-'*50}")
    for name in models_config:
        scores = cv_results[name]
        verdict = " << BEST" if name == best_name else ""
        print(f"  {name:<18} {scores.mean():>10.4f} {scores.std():>8.4f}  {verdict}")

    print(f"\n  Best model saved to: {model_path}")
    print(f"  Charts: 10_model_comparison.png, 11_optuna_optimization.png, 12_roc_curves.png")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def _setup_dark_axes(ax, title):
    """Apply dark theme to matplotlib axes."""
    ax.set_facecolor(CARD_COLOR)
    ax.set_title(title, color=TEXT_COLOR, fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=0.5)


def plot_model_comparison(cv_results, output_dir):
    """Grouped bar chart: AUC per model per fold."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
    _setup_dark_axes(ax, "5-Fold Cross-Validation AUC Comparison")

    models = list(cv_results.keys())
    n_folds = len(list(cv_results.values())[0])
    x = np.arange(n_folds)
    width = 0.25

    for i, name in enumerate(models):
        scores = cv_results[name]
        bars = ax.bar(x + i * width, scores, width, label=name,
                      color=COLORS[name], alpha=0.85, edgecolor="white", linewidth=0.5)
        # Add value labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{score:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color=TEXT_COLOR, fontweight="bold")

    # Mean lines
    for name in models:
        mean_val = cv_results[name].mean()
        ax.axhline(y=mean_val, color=COLORS[name], linestyle="--",
                   alpha=0.6, linewidth=1)
        ax.text(n_folds - 0.5 + len(models) * width, mean_val,
                f"  {name}: {mean_val:.3f}",
                va="center", fontsize=8, color=COLORS[name], fontweight="bold")

    ax.set_xlabel("Fold", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("ROC-AUC Score", color=TEXT_COLOR, fontsize=11)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"Fold {i+1}" for i in range(n_folds)])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9, facecolor=CARD_COLOR,
              edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    plt.tight_layout()
    path = output_dir / "10_model_comparison.png"
    fig.savefig(path, dpi=150, facecolor=BG_COLOR)
    plt.close(fig)
    print(f"   Saved: {path}")


def plot_optuna_history(study_histories, output_dir):
    """Optimization history curves for each model."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
    _setup_dark_axes(ax, "Optuna Optimization History")

    for name, history in study_histories.items():
        # Compute running best
        running_best = []
        best_so_far = -1
        for val in history:
            best_so_far = max(best_so_far, val)
            running_best.append(best_so_far)

        trials = range(1, len(history) + 1)
        ax.plot(trials, history, ".", color=COLORS[name], alpha=0.25, markersize=4)
        ax.plot(trials, running_best, "-", color=COLORS[name], linewidth=2,
                label=f"{name} (best: {max(history):.4f})")

    ax.set_xlabel("Trial #", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("ROC-AUC Score", color=TEXT_COLOR, fontsize=11)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="lower right", fontsize=9, facecolor=CARD_COLOR,
              edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    plt.tight_layout()
    path = output_dir / "11_optuna_optimization.png"
    fig.savefig(path, dpi=150, facecolor=BG_COLOR)
    plt.close(fig)
    print(f"   Saved: {path}")


def plot_roc_curves(final_models, X, y, cv, output_dir):
    """Overlaid ROC curves for all 3 models (mean + std band)."""
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG_COLOR)
    _setup_dark_axes(ax, "ROC Curves - Model Comparison")

    mean_fpr = np.linspace(0, 1, 100)

    for name, clf in final_models.items():
        tprs = []
        aucs = []

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf_clone = clone_model(clf)
            clf_clone.fit(X_train, y_train)
            y_prob = clf_clone.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            aucs.append(roc_auc_score(y_test, y_prob))

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_tpr = np.std(tprs, axis=0)

        ax.plot(mean_fpr, mean_tpr, color=COLORS[name], linewidth=2,
                label=f"{name} (AUC = {mean_auc:.3f})")
        ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                        color=COLORS[name], alpha=0.12)

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "--", color="#555555", linewidth=1, label="Random (0.500)")

    ax.set_xlabel("False Positive Rate", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("True Positive Rate", color=TEXT_COLOR, fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.legend(loc="lower right", fontsize=10, facecolor=CARD_COLOR,
              edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    plt.tight_layout()
    path = output_dir / "12_roc_curves.png"
    fig.savefig(path, dpi=150, facecolor=BG_COLOR)
    plt.close(fig)
    print(f"   Saved: {path}")


def clone_model(clf):
    """Create a fresh clone of a model with the same params."""
    from sklearn.base import clone as sk_clone
    return sk_clone(clf)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare RF, XGBoost, LightGBM with Optuna tuning")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to training data CSV (auto-detects)")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of Optuna trials per model (default: 50)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                        help="Output directory")
    args = parser.parse_args()

    # Auto-detect data file
    if args.data:
        data_path = Path(args.data)
    elif GLOBAL_DATA.exists():
        data_path = GLOBAL_DATA
        print(f"   Using global training data: {data_path}")
    else:
        data_path = DEFAULT_DATA
        print(f"   Using default training data: {data_path}")

    if not data_path.exists():
        sys.exit(f"ERROR: Training data not found: {data_path}\n"
                 f"  Run the ML pipeline first: python main.py --train")

    run_comparison(data_path, n_trials=args.trials, output_dir=args.output)


if __name__ == "__main__":
    main()
