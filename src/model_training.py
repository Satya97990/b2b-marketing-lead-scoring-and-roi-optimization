"""
Module: model_training.py
Description:
    Trains, evaluates, and compares three predictive models for B2B Marketing Lead Scoring:
    1. Logistic Regression (Interpretable baseline for marketing stakeholders)
    2. Random Forest Classifier (Non-linear ensemble capturing interaction effects)
    3. HistGradientBoosting Classifier (High-performance gradient boosted decision trees)

    Evaluates models on ROC-AUC, PR-AUC (Average Precision), F1-Score, Accuracy, and Brier Score
    (probability calibration quality). Saves the best pipeline and feature importances.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve
)
from sklearn.inspection import permutation_importance
from src.preprocessing import build_sklearn_preprocessor


CATEGORICAL_FEATURES = [
    "Lead_Origin_Channel",
    "Buyer_Persona",
    "Company_Size",
    "Industry_Vertical",
    "Last_Notable_Activity"
]

NUMERICAL_FEATURES = [
    "Acquisition_Cost_USD",
    "Total_Web_Visits",
    "Total_Time_Spent_Sec",
    "Page_Views_Per_Visit",
    "Downloaded_Whitepaper",
    "Pricing_Page_Visited",
    "Requested_Product_Demo",
    "Email_Open_Rate",
    "Avg_Time_Per_Visit_Sec",
    "High_Intent_Action_Count",
    "Digital_Engagement_Index",
    "Deal_Value_to_CAC_Ratio"
]


def train_and_evaluate_models(
    df: pd.DataFrame,
    model_Save_path: str = "models/lead_scoring_pipeline.pkl",
    metrics_save_path: str = "models/model_metrics.json"
) -> dict:
    """
    Splits the dataset, trains candidate lead scoring models, logs evaluation metrics,
    computes permutation feature importances, and serializes the winning model pipeline.
    """
    X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = df["Converted"].values

    # Stratified 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=10, min_samples_leaf=10, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting (HistGB)": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_depth=6, min_samples_leaf=20, random_state=42
        )
    }

    metrics_summary = {}
    curves_data = {}
    trained_pipelines = {}

    best_model_name = None
    best_roc_auc = -1.0

    for name, estimator in candidate_models.items():
        preprocessor = build_sklearn_preprocessor(CATEGORICAL_FEATURES, NUMERICAL_FEATURES)
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator)
        ])

        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.50).astype(int)

        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_prob)

        metrics_summary[name] = {
            "ROC_AUC": round(float(roc_auc), 4),
            "PR_AUC": round(float(pr_auc), 4),
            "F1_Score": round(float(f1), 4),
            "Precision": round(float(precision), 4),
            "Recall": round(float(recall), 4),
            "Accuracy": round(float(acc), 4),
            "Brier_Score": round(float(brier), 4)
        }

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
        curves_data[name] = {
            "fpr": fpr,
            "tpr": tpr,
            "prec_curve": prec_curve,
            "rec_curve": rec_curve,
            "y_prob": y_prob,
            "y_test": y_test
        }

        trained_pipelines[name] = pipeline
        print(f"[MODEL] {name:<28} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | F1: {f1:.4f} | Brier: {brier:.4f}")

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_name = name

    best_pipeline = trained_pipelines[best_model_name]
    print(f"[INFO] Selected Best Model: {best_model_name} (ROC-AUC = {best_roc_auc:.4f})")

    # Compute Permutation Feature Importance on Test Set for transparent explainability
    perm_result = permutation_importance(
        best_pipeline, X_test, y_test, n_repeats=8, random_state=42, scoring="roc_auc", n_jobs=-1
    )
    feature_importance_df = pd.DataFrame({
        "Feature": CATEGORICAL_FEATURES + NUMERICAL_FEATURES,
        "Importance_Mean": perm_result.importances_mean,
        "Importance_Std": perm_result.importances_std
    }).sort_values(by="Importance_Mean", ascending=False).reset_index(drop=True)

    # Save best model pipeline and metrics JSON
    os.makedirs(os.path.dirname(model_Save_path), exist_ok=True)
    joblib.dump(best_pipeline, model_Save_path)

    json_payload = {
        "best_model": best_model_name,
        "models": metrics_summary,
        "top_features": feature_importance_df.to_dict(orient="records")
    }
    with open(metrics_save_path, "w") as f:
        json.dump(json_payload, f, indent=4)

    return {
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "metrics_summary": metrics_summary,
        "curves_data": curves_data,
        "feature_importance_df": feature_importance_df
    }
