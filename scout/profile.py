"""Profiler / EDA agent logic (syllabus modules 1-3).

Computes everything *from the data it is given*. In experiments it is only ever called on the
training fold, so the resulting preprocessing plan cannot leak test information.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest


def _split_types(X: pd.DataFrame) -> Dict[str, List[str]]:
    num, cat, dt = [], [], []
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_bool_dtype(s):
            num.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            num.append(c)
        elif pd.api.types.is_datetime64_any_dtype(s):
            dt.append(c)
        else:
            if s.dtype == object or str(s.dtype) in ("str", "string", "category"):
                nm = str(c).lower()
                if any(k in nm for k in ("date", "time", "timestamp")):
                    parsed = pd.to_datetime(s, errors="coerce")
                    if parsed.notna().mean() > 0.8:
                        dt.append(c)
                        continue
            cat.append(c)
    return {"numeric": num, "categorical": cat, "datetime": dt}


def missingness_mechanism(X: pd.DataFrame, y: Optional[pd.Series], col: str, alpha: float = 0.01,
                          max_probe: int = 25) -> str:
    """Heuristic MCAR/MAR diagnostic for one column (syllabus: MCAR, MAR, MNAR).

    Tests whether the missing-indicator of `col` is associated with other observed columns / the target
    (Mann-Whitney U for numeric, chi-square for categorical, Bonferroni-corrected).
    * association found  -> "MAR-suspected"
    * none found         -> "MCAR-not-rejected"
    MNAR cannot be identified from observed data alone; it is reported as a caveat, never as a finding.
    """
    m = X[col].isna()
    if m.sum() < 5 or (~m).sum() < 5:
        return "insufficient-data"
    pvals = []
    others = [c for c in X.columns if c != col][:max_probe]
    frame = X[others].copy()
    if y is not None:
        frame["__y__"] = y.values
    for c in frame.columns:
        s = frame[c]
        try:
            if pd.api.types.is_numeric_dtype(s):
                a, b = s[m].dropna(), s[~m].dropna()
                if len(a) >= 5 and len(b) >= 5 and (a.nunique() > 1 or b.nunique() > 1):
                    pvals.append(stats.mannwhitneyu(a, b).pvalue)
            else:
                ct = pd.crosstab(m, s.astype(object).fillna("NA"))
                if ct.shape[0] == 2 and ct.shape[1] > 1:
                    pvals.append(stats.chi2_contingency(ct)[1])
        except Exception:
            continue
    if not pvals:
        return "insufficient-data"
    return "MAR-suspected" if min(pvals) < alpha / len(pvals) else "MCAR-not-rejected"


def profile_dataframe(X: pd.DataFrame, y: Optional[pd.Series] = None, seed: int = 0,
                      with_outliers: bool = True) -> Dict[str, Any]:
    n, d = X.shape
    types = _split_types(X)
    num, cat = types["numeric"], types["categorical"]

    # ---- structure / duplicates / constants / ids
    dup = int(X.duplicated().sum())
    nun = X.nunique(dropna=True)
    constant = [c for c in X.columns if nun[c] <= 1]
    id_like = [c for c in cat if n > 0 and nun[c] / n > 0.95]
    # ---- missingness
    miss_frac = X.isna().mean()
    miss_cols = {c: float(miss_frac[c]) for c in X.columns if miss_frac[c] > 0}
    mechanisms = {c: missingness_mechanism(X, y, c) for c in miss_cols if miss_cols[c] < 0.9}
    # ---- descriptive statistics (central tendency, dispersion, skewness, kurtosis)
    desc, skew, kurt = {}, {}, {}
    for c in num:
        s = pd.to_numeric(X[c], errors="coerce").astype(float)
        s_ = s.dropna()
        if len(s_) > 3 and s_.nunique() > 1:
            skew[c] = float(stats.skew(s_))
            kurt[c] = float(stats.kurtosis(s_))
            q1, q3 = s_.quantile([0.25, 0.75])
            desc[c] = dict(mean=float(s_.mean()), median=float(s_.median()), std=float(s_.std()),
                           iqr=float(q3 - q1), min=float(s_.min()), max=float(s_.max()))
    # ---- outliers: z-score, IQR, Isolation Forest (syllabus module 2)
    out_iqr, out_z = {}, {}
    for c in desc:
        s = pd.to_numeric(X[c], errors="coerce").dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        out_iqr[c] = float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean())
        sd = s.std()
        out_z[c] = float((np.abs((s - s.mean()) / sd) > 3).mean()) if sd > 0 else 0.0
    iso_frac = 0.0
    if with_outliers and len(desc) >= 2 and n >= 50:
        M = X[list(desc)].apply(pd.to_numeric, errors="coerce")
        M = M.fillna(M.median()).to_numpy(dtype=float)
        if len(M) > 5000:
            M = M[np.random.RandomState(seed).choice(len(M), 5000, replace=False)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            iso = IsolationForest(n_estimators=100, contamination="auto", random_state=seed).fit(M)
            iso_frac = float((iso.predict(M) == -1).mean())
    # ---- categorical cardinality
    card = {c: int(nun[c]) for c in cat}
    high_card = [c for c in cat if card[c] > 15 and c not in id_like]
    # ---- correlation structure
    max_corr, n_corr_pairs = 0.0, 0
    if len(desc) >= 2:
        C = X[list(desc)].apply(pd.to_numeric, errors="coerce").corr().abs().to_numpy()
        iu = np.triu_indices_from(C, k=1)
        vals = C[iu][~np.isnan(C[iu])]
        if len(vals):
            max_corr, n_corr_pairs = float(vals.max()), int((vals > 0.9).sum())
    # ---- target / bias
    target = {}
    if y is not None:
        vc = y.value_counts()
        target = dict(n_classes=int(len(vc)), counts={str(k): int(v) for k, v in vc.items()},
                      imbalance_ratio=float(vc.min() / vc.max()),
                      minority_fraction=float(vc.min() / vc.sum()))
    bias_flags = []
    if target and target["minority_fraction"] < 0.10:
        bias_flags.append(f"class imbalance: minority {target['minority_fraction']:.1%}")
    for c in cat:
        top = X[c].value_counts(normalize=True, dropna=True)
        if len(top) and top.iloc[0] > 0.95:
            bias_flags.append(f"dominant level (>95%) in '{c}'")

    # ---- transparent data-quality score (100 minus capped penalties)
    tot_miss = float(X.isna().to_numpy().mean()) if n * d else 0.0
    pen = dict(
        missing=min(30.0, 100 * tot_miss * 1.5),
        duplicates=min(10.0, 100 * dup / max(n, 1) * 2),
        high_cardinality=min(10.0, 2.5 * len(high_card)),
        outliers=min(15.0, 100 * (np.mean(list(out_iqr.values())) if out_iqr else 0) * 1.5),
        imbalance=(min(20.0, 20 * (1 - target["imbalance_ratio"]) if target else 0.0)),
        constants=min(5.0, 2.5 * len(constant)),
    )
    quality = float(max(0.0, 100 - sum(pen.values())))
    return dict(
        n_rows=n, n_cols=d, types=types, duplicates=dup, constant=constant, id_like=id_like,
        total_missing_frac=tot_miss, missing=miss_cols, missing_mechanism=mechanisms,
        descriptive=desc, skew=skew, kurtosis=kurt, outlier_iqr=out_iqr, outlier_z=out_z,
        outlier_isolation_forest=iso_frac, cardinality=card, high_cardinality=high_card,
        max_abs_corr=max_corr, n_corr_pairs_gt_09=n_corr_pairs, target=target, bias_flags=bias_flags,
        quality_penalties=pen, quality_score=quality,
    )
