"""Evaluation + explanation agents. Multi-metric by design (accuracy alone is never reported)."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, mean_squared_error, r2_score, roc_auc_score)


def majority_baseline(y_train, y_test) -> Dict[str, float]:
    """Majority-class accuracy computed on the *same test split* (fixes the baseline error in v1)."""
    maj = pd.Series(y_train).value_counts().idxmax()
    pred = np.full(len(y_test), maj)
    return dict(acc=float(accuracy_score(y_test, pred)), bal_acc=float(balanced_accuracy_score(y_test, pred)))


def classification_metrics(y_true, y_pred, proba: Optional[np.ndarray], classes) -> Dict[str, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = dict(accuracy=accuracy_score(y_true, y_pred), balanced_acc=balanced_accuracy_score(y_true, y_pred),
               macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0))
    classes = np.asarray(classes)
    if proba is not None and np.isfinite(proba).all():
        try:
            if len(classes) == 2:
                pos = (y_true == classes[1]).astype(int)
                out["roc_auc"] = roc_auc_score(pos, proba[:, 1])
                out["pr_auc"] = average_precision_score(pos, proba[:, 1])
                out["brier"] = brier_score_loss(pos, proba[:, 1])
            else:
                present = np.isin(classes, np.unique(y_true))
                P = proba[:, present]
                P = P / P.sum(1, keepdims=True)
                out["roc_auc"] = roc_auc_score(y_true, P, multi_class="ovr", labels=classes[present], average="macro")
            out["log_loss"] = log_loss(y_true, np.clip(proba, 1e-12, 1), labels=classes)
        except Exception:
            pass
    return {k: float(v) for k, v in out.items()}


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    return dict(r2=float(r2_score(y_true, y_pred)), rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))))


def confusion(y_true, y_pred, classes):
    return confusion_matrix(y_true, y_pred, labels=classes)


def explain(model, X_test: pd.DataFrame, y_test, task="classification", n_repeats=5, seed=0, top=10) -> pd.DataFrame:
    """Permutation importance on the held-out set (model-agnostic, on *raw* columns -> readable names).
    Associational, never causal."""
    scoring = "balanced_accuracy" if task == "classification" else "r2"
    r = permutation_importance(model, X_test, y_test, scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=1)
    df = pd.DataFrame(dict(feature=X_test.columns, importance=r.importances_mean, std=r.importances_std))
    return df.sort_values("importance", ascending=False).head(top).reset_index(drop=True)
