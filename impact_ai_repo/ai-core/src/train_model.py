#!/usr/bin/env python3
"""
train_model.py

Train models (Logistic Regression, Random Forest, Gradient Boosting) on a features file.
Uses an explicit target column from the features file (for example breaking_change).

Exports:
  - feature_columns.json       numeric feature names used for training
  - model.joblib               best model (by ROC AUC)
  - training_report.json       metrics for all models
  - predictions.csv            full dataset predictions for the best model
  - pr_LogisticRegression.png  Precision Recall curve on test set
  - pr_RandomForest.png        Precision Recall curve on test set
  - pr_GradientBoosting.png    Precision Recall curve on test set

Usage:
    python train_model.py \
      --input analysis_outputs/curated_clean/features.csv \
      --target breaking_change \
      --outdir analysis_outputs \
      --tag curated_clean \
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)

# -------------------------------
# Helper: evaluate a model
# -------------------------------
def evaluate(model, X_test, y_test):
    """
    Evaluate a trained model on the test set.
    Returns metrics and probabilities.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        raw = model.decision_function(X_test)
        probs = 1.0 / (1.0 + np.exp(-raw))

    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, probs))
    except Exception:
        metrics["roc_auc"] = None

    # PR AUC for reference
    try:
        pr_prec, pr_rec, _ = precision_recall_curve(y_test, probs)
        metrics["pr_auc"] = float(auc(pr_rec, pr_prec))
    except Exception:
        metrics["pr_auc"] = None

    return metrics, probs


# -------------------------------
# Main training function
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train ML models for API impact analysis")
    parser.add_argument("--input", required=True, help="Path to features CSV or parquet file")
    parser.add_argument("--target", required=True, help="Name of the label column in the features file")
    parser.add_argument(
        "--outdir",
        default="./analysis_outputs",
        help="Base directory to store trained models and outputs",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional dataset tag to create a subfolder inside outdir",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    np.random.seed(args.seed)

    base_outdir = Path(args.outdir)

    # If a tag is provided, write into outdir/tag
    if args.tag:
        outdir = base_outdir / args.tag
    else:
        outdir = base_outdir

    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------
    # Load input data
    # -------------------
    if args.input.endswith(".csv"):
        df = pd.read_csv(args.input)
    else:
        df = pd.read_parquet(args.input)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in dataset")

    y = df[args.target].astype(int)
    X = df.drop(columns=[args.target])

    # Keep only numeric columns for ML
    numeric_cols = [c for c in X.columns if X[c].dtype.kind in "biuf"]
    if not numeric_cols:
        raise SystemExit("No numeric feature columns detected. Check features file and target column")

    X = X[numeric_cols]

    # Save column list for runtime alignment
    with open(outdir / "feature_columns.json", "w") as fh:
        json.dump(numeric_cols, fh, indent=2)

    # -------------------
    # Split into train and test
    # -------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y if len(set(y)) > 1 else None,
    )

    # -------------------
    # Create candidate models
    # -------------------
    base_preproc = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    candidates = {
        "LogisticRegression": Pipeline(
            [
                ("pre", base_preproc),
                ("clf", LogisticRegression(max_iter=1000, random_state=args.seed)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("pre", base_preproc),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=args.seed)),
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                ("pre", base_preproc),
                ("clf", GradientBoostingClassifier(n_estimators=200, random_state=args.seed)),
            ]
        ),
    }

    results = {}
    best_model = None
    best_name = None
    best_auc = -1.0

    # Matplotlib for PR curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # -------------------
    # Train and evaluate each model
    # -------------------
    for name, model in candidates.items():
        print(f"Training {name} ...")
        t0 = time.time()
        model.fit(X_train, y_train)
        metrics, probs = evaluate(model, X_test, y_test)
        metrics["train_time"] = float(time.time() - t0)
        results[name] = metrics

        # Track best model by ROC AUC
        auc_score = metrics.get("roc_auc") or -1.0
        if auc_score > best_auc:
            best_auc = auc_score
            best_model = model
            best_name = name

        # Precision Recall curve on test set
        try:
            pr_prec, pr_rec, _ = precision_recall_curve(y_test, probs)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(pr_rec, pr_prec, label=name)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"Precision Recall curve for {name}")
            ax.grid(True, linestyle=":", linewidth=0.5)
            ax.legend(loc="best")

            png_path = outdir / f"pr_{name}.png"
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved PR curve to {png_path}")
        except Exception as e:
            print(f"Warning. Failed to generate PR curve for {name}. {e}")

    # -------------------
    # Save best model
    # -------------------
    if best_model is None:
        raise SystemExit("No model trained successfully. Check data and labels")

    joblib.dump(best_model, outdir / "model.joblib")
    print(f"Saved best model {best_name} to {outdir / 'model.joblib'}")

    # -------------------
    # Generate predictions.csv on full dataset for best model
    # -------------------
    try:
        print("Generating predictions on full dataset for best model")
        X_full = df[numeric_cols]

        if hasattr(best_model, "predict_proba"):
            probs_full = best_model.predict_proba(X_full)[:, 1]
        else:
            raw_full = best_model.decision_function(X_full)
            probs_full = 1.0 / (1.0 + np.exp(-raw_full))

        preds_full = (probs_full >= 0.5).astype(int)

        preds_df = pd.DataFrame(
            {
                "ace_id": df.get("ace_id", pd.Series([None] * len(df))),
                "probability": probs_full,
                "label": y,
                "predicted": preds_full,
                "model": best_name,
            }
        )

        preds_path = outdir / "predictions.csv"
        preds_df.to_csv(preds_path, index=False)
        print(f"Saved predictions to {preds_path}")
    except Exception as e:
        print(f"Warning. Failed to generate predictions.csv. {e}")

    # -------------------
    # Save training summary as training_report.json
    # -------------------
    summary = {
        "best_model": best_name,
        "best_auc": float(best_auc) if best_auc is not None else None,
        "results": results,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "seed": int(args.seed),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    report_path = outdir / "training_report.json"
    with open(report_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Training complete. Best model {best_name} with AUC {best_auc:.3f}")
    print(f"Artifacts written to {outdir}")


if __name__ == "__main__":
    main()
