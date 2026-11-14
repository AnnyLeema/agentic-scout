"""Preprocessing agent (syllabus module 2 + 4).

`plan_preprocessing(profile)` turns an EDA profile into an explicit, human-readable plan (each decision
carries its reason -> audit trail). `build_preprocessor(plan, scale_sensitive)` turns the plan into an
sklearn transformer that is fitted on training data only.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, PowerTransformer, StandardScaler, TargetEncoder)


class FrameCleaner(BaseEstimator, TransformerMixin):
    """Normalises dtypes: numerics -> float, categoricals -> object with np.nan, dates -> numeric parts."""

    def __init__(self, numeric=None, categorical=None, datetime=None):
        self.numeric, self.categorical, self.datetime = numeric, categorical, datetime

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        for c in self.numeric or []:
            out[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
        for c in self.categorical or []:
            s = X[c].astype(object)
            out[c] = s.where(s.notna(), np.nan).map(lambda v: v if isinstance(v, float) else str(v))
        for c in self.datetime or []:
            d = pd.to_datetime(X[c], errors="coerce")
            out[f"{c}__year"], out[f"{c}__month"] = d.dt.year.astype(float), d.dt.month.astype(float)
            out[f"{c}__dow"] = d.dt.dayofweek.astype(float)
        return out


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clips to Tukey fences (Q1-1.5 IQR, Q3+1.5 IQR) learned on training data (IQR outlier treatment)."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lo_, self.hi_ = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


class TopK(BaseEstimator, TransformerMixin):
    """Filter-method feature selection (mutual information) with k capped at n_features."""

    def __init__(self, k=100, task="classification", seed=0):
        self.k, self.task, self.seed = k, task, seed

    def fit(self, X, y):
        k = min(self.k, X.shape[1])
        if self.task == "classification":
            fn = lambda a, b: mutual_info_classif(a, b, random_state=self.seed)
        else:
            from sklearn.feature_selection import mutual_info_regression
            fn = lambda a, b: mutual_info_regression(a, b, random_state=self.seed)
        self.sel_ = SelectKBest(fn, k=k).fit(X, y)
        return self

    def transform(self, X):
        return self.sel_.transform(X)


# --------------------------------------------------------------------------------------------------
def plan_preprocessing(profile: Dict[str, Any], task: str = "classification", adaptive: bool = True) -> Dict[str, Any]:
    """Rule-based, deterministic plan. adaptive=False gives the fixed generic baseline used in ablations."""
    types = profile["types"]
    drop = set(profile["constant"]) | set(profile["id_like"])
    reasons: List[str] = []
    if profile["constant"]:
        reasons.append(f"dropped constant columns {profile['constant']}")
    if profile["id_like"]:
        reasons.append(f"dropped ID-like columns {profile['id_like']} (unique ratio > 0.95)")
    for c, f in profile["missing"].items():
        if f > 0.6:
            drop.add(c)
            reasons.append(f"dropped '{c}': {f:.0%} missing (> 60%)")
    num = [c for c in types["numeric"] if c not in drop]
    cat = [c for c in types["categorical"] if c not in drop]
    dt = [c for c in types["datetime"] if c not in drop]

    plan = dict(numeric=num, categorical=cat, datetime=dt, dropped=sorted(drop), adaptive=adaptive,
                impute_num="median", add_indicator=False, skewed=[], winsorize=False,
                high_card=[], low_card=[], pca=None, select_k=None, class_weight=False, reasons=reasons)
    if not adaptive:
        plan["low_card"] = cat
        reasons.append("generic plan: median/mode imputation, one-hot, standardisation")
        return plan

    # --- missing values: MCAR -> simple imputation; MAR -> MICE + missing indicators (module 2)
    num_miss = {c: profile["missing"][c] for c in num if c in profile["missing"]}
    mar = [c for c in num_miss if profile["missing_mechanism"].get(c) == "MAR-suspected"]
    total_num_miss = max(num_miss.values()) if num_miss else 0.0
    if mar and len(num) >= 3 and profile["n_rows"] <= 20000:
        plan["impute_num"] = "mice"
        reasons.append(f"MAR suspected for {mar[:5]} -> MICE (IterativeImputer) for numerics")
    elif num_miss:
        reasons.append("missingness not associated with other variables (MCAR not rejected) -> median imputation")
    if num_miss and (mar or total_num_miss > 0.05):
        plan["add_indicator"] = True
        reasons.append("added missing-value indicators (informative-missingness safeguard; MNAR cannot be tested)")
    # --- skew -> power transform (module 2: log / sqrt / power)
    plan["skewed"] = [c for c in num if abs(profile["skew"].get(c, 0.0)) > 1.0]
    if plan["skewed"]:
        reasons.append(f"|skew|>1 for {len(plan['skewed'])} numeric columns -> Yeo-Johnson power transform")
    # --- outliers
    mean_iqr = float(np.mean([profile["outlier_iqr"].get(c, 0) for c in num])) if num else 0.0
    if mean_iqr > 0.02:
        plan["winsorize"] = True
        reasons.append(f"mean IQR-outlier rate {mean_iqr:.1%} > 2% -> Tukey-fence clipping (scale-sensitive models)")
    # --- categorical encoding
    hc = [c for c in cat if profile["cardinality"].get(c, 0) > 15]
    plan["high_card"], plan["low_card"] = hc, [c for c in cat if c not in hc]
    if hc:
        reasons.append(f"high-cardinality {hc[:4]} -> cross-fitted target encoding; others one-hot")
    # --- dimensionality (module 4: filter + PCA)
    est_dim = len(num) + sum(min(profile["cardinality"].get(c, 0), 15) for c in plan["low_card"]) + len(hc)
    if est_dim > 150:
        plan["select_k"] = 100
        reasons.append(f"~{est_dim} features -> filter selection (mutual information, k=100)")
    if profile["n_corr_pairs_gt_09"] >= 3 and len(num) > 10:
        plan["pca"] = 0.95
        reasons.append("many |r|>0.9 pairs -> PCA (95% variance) for scale-sensitive models")
    # --- imbalance
    tgt = profile.get("target", {})
    if task == "classification" and tgt and tgt["minority_fraction"] < 0.20:
        plan["class_weight"] = True
        reasons.append(f"minority {tgt['minority_fraction']:.1%} -> class-weighted learning + balanced-accuracy objective")
    return plan


def build_preprocessor(plan: Dict[str, Any], scale_sensitive: bool, task: str = "classification", seed: int = 0):
    adaptive = plan["adaptive"]
    steps = []
    cleaner = FrameCleaner(plan["numeric"], plan["categorical"], plan["datetime"])
    dt_cols = [f"{c}__{p}" for c in plan["datetime"] for p in ("year", "month", "dow")]
    num_all = plan["numeric"] + dt_cols
    skewed = [c for c in plan["skewed"] if c in num_all] if (adaptive and scale_sensitive) else []
    normal = [c for c in num_all if c not in skewed]

    def num_pipe(use_power):
        if plan["impute_num"] == "mice":
            imp = IterativeImputer(max_iter=5, random_state=seed, add_indicator=plan["add_indicator"],
                                   skip_complete=True)
        else:
            imp = SimpleImputer(strategy="median", add_indicator=plan["add_indicator"] if adaptive else False)
        st = [("imp", imp)]
        if adaptive and plan["winsorize"] and scale_sensitive:
            st.append(("win", Winsorizer()))
        if use_power:
            st.append(("power", PowerTransformer(method="yeo-johnson", standardize=True)))
        elif scale_sensitive:
            st.append(("scale", StandardScaler()))
        return Pipeline(st)

    trans = []
    if normal:
        trans.append(("num", num_pipe(False), normal))
    if skewed:
        trans.append(("skew", num_pipe(True), skewed))
    if plan["low_card"]:
        trans.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                                            min_frequency=0.005))]), plan["low_card"]))
    if plan["high_card"]:
        te = TargetEncoder(random_state=seed, target_type="auto")
        trans.append(("hcat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("te", te)]),
                      plan["high_card"]))
    ct = ColumnTransformer(trans, remainder="drop", sparse_threshold=0.0, verbose_feature_names_out=False)
    steps = [("clean", cleaner), ("ct", ct)]
    if adaptive and plan["select_k"]:
        steps.append(("select", TopK(plan["select_k"], task, seed)))
    if adaptive and plan["pca"] and scale_sensitive:
        steps.append(("pca", PCA(n_components=plan["pca"], random_state=seed)))
    return Pipeline(steps)
