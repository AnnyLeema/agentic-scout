"""Dataset registry: scikit-learn bundled data (offline) + OpenML (needs internet).

All datasets are public benchmark data. Nothing here is synthetic.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn import datasets as skd
from sklearn.preprocessing import LabelEncoder

CACHE = os.environ.get("SCOUT_CACHE", os.path.join(os.path.dirname(__file__), "..", "data_cache"))

# ---- offline (bundled with scikit-learn) ------------------------------------
_BUILTIN = {
    "iris": skd.load_iris,
    "wine": skd.load_wine,
    "breast_cancer": skd.load_breast_cancer,   # UCI WDBC
    "digits": skd.load_digits,                 # UCI optical digits (8x8)
}

# ---- OpenML datasets used in the syllabus labs / common benchmarks ----------
# (IDs should be re-verified on openml.org before the final run.)
NAMED_OPENML: Dict[str, int] = {
    "titanic": 40945,            # Lab 1
    "wine_quality_white": 40498, # Lab 3
    "creditcard_fraud": 1597,    # Lab 4 (284k rows -> subsampled)
    "mnist_784": 554,            # Lab 5 (70k rows -> subsampled)
    "credit_g": 31,
    "diabetes_pima": 37,
    "bank_marketing": 1461,
    "adult": 1590,
}


# ---- PMLB (Penn Machine Learning Benchmarks, Romano et al. 2021) -------------
# Hosted on GitHub (git-LFS): fetched via media.githubusercontent.com, cached locally.
PMLB_URL = "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/{n}/{n}.tsv.gz"
# Curated list of real-world datasets that download cleanly (synthetic generators such as waveform/led/monk excluded).
PMLB_CURATED: List[str] = [
    "haberman", "bupa", "ecoli", "ionosphere", "dermatology", "saheart", "spectf", "penguins", "cars",
    "analcatdata_germangss", "irish", "profb", "balance_scale", "vehicle", "tic_tac_toe", "tokyo1",
    "analcatdata_dmft", "yeast", "wine_quality_red", "car_evaluation", "mfeat_zernike", "mfeat_morphological",
    "segmentation", "hypothyroid", "splice", "kr_vs_kp", "allbp", "spambase", "page_blocks", "phoneme",
    "dis", "allhypo", "churn", "wine_quality_white",          # spare (used only if more than 30 requested)
]


def _load_pmlb(name: str) -> Tuple[pd.DataFrame, pd.Series]:
    import urllib.request
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"pmlb_{name}.tsv.gz")
    if not os.path.exists(path) or os.path.getsize(path) < 400:
        urllib.request.urlretrieve(PMLB_URL.format(n=name), path)
    df = pd.read_csv(path, sep="\t", compression="gzip")
    if "target" not in df.columns or len(df) < 50:
        raise ValueError(f"PMLB download for {name} looks wrong (git-LFS pointer?)")
    return df.drop(columns=["target"]), _encode_target(df["target"])


def _encode_target(y) -> pd.Series:
    return pd.Series(LabelEncoder().fit_transform(pd.Series(y).astype(str)), name="target")


def stratified_subsample(X: pd.DataFrame, y: pd.Series, max_rows: int, seed: int = 0):
    if len(X) <= max_rows:
        return X.reset_index(drop=True), y.reset_index(drop=True)
    from sklearn.model_selection import train_test_split
    try:
        Xs, _, ys, _ = train_test_split(X, y, train_size=max_rows, stratify=y, random_state=seed)
    except ValueError:
        Xs, _, ys, _ = train_test_split(X, y, train_size=max_rows, random_state=seed)
    return Xs.reset_index(drop=True), ys.reset_index(drop=True)


def load_dataset(name: str, max_rows: int = 5000, seed: int = 0) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X DataFrame, y integer Series) for classification datasets."""
    if name in _BUILTIN:
        b = _BUILTIN[name]()
        X = pd.DataFrame(b.data, columns=[str(c) for c in b.feature_names])
        y = pd.Series(b.target, name="target")
        return stratified_subsample(X, y, max_rows, seed)
    if name.startswith("csv:"):               # csv:/path/file.csv:TargetColumn  (e.g. your retail dataset)
        _, path, target = name.split(":", 2)
        df = pd.read_csv(path)
        y = _encode_target(df[target])
        X = df.drop(columns=[target])
        return stratified_subsample(X, y, max_rows, seed)
    if name.startswith("pmlb:"):
        X, y = _load_pmlb(name.split(":", 1)[1])
        return stratified_subsample(X, y, max_rows, seed)
    if name.startswith("openml:"):
        data_id = int(name.split(":")[1])
    elif name in NAMED_OPENML:
        data_id = NAMED_OPENML[name]
    else:
        raise KeyError(f"Unknown dataset {name}")
    from sklearn.datasets import fetch_openml
    os.makedirs(CACHE, exist_ok=True)
    bunch = fetch_openml(data_id=data_id, as_frame=True, data_home=CACHE, parser="auto")
    X, y = bunch.data.copy(), _encode_target(bunch.target)
    for c in X.columns:                      # keep category dtypes as object for the profiler
        if str(X[c].dtype) == "category":
            X[c] = X[c].astype(object)
    return stratified_subsample(X, y, max_rows, seed)


def list_cc18(max_instances: int = 20000, max_features: int = 500, min_instances: int = 150,
              limit: int | None = None) -> List[str]:
    """OpenML-CC18 (72 curated classification tasks) filtered by size. Needs internet."""
    import openml
    suite = openml.study.get_suite(99)
    ids = list(suite.data)
    try:
        df = openml.datasets.list_datasets(data_id=ids, output_format="dataframe")
    except TypeError:                         # newer openml versions
        df = openml.datasets.list_datasets(data_id=ids)
    df = df[(df.NumberOfInstances <= max_instances) & (df.NumberOfInstances >= min_instances)
            & (df.NumberOfFeatures <= max_features) & (df.NumberOfClasses >= 2)]
    df = df.sort_values("NumberOfInstances")
    out = [f"openml:{int(i)}" for i in df["did"].tolist()]
    return out[:limit] if limit else out


def resolve(spec: str, **kw) -> List[str]:
    """'builtin' | 'cc18' | 'cc18:30' | 'labs' | comma-separated names."""
    if spec == "builtin":
        return list(_BUILTIN)
    if "," in spec:
        return [x.strip() for x in spec.split(",") if x.strip()]
    if spec.startswith("pmlb"):                # 'pmlb' (all curated) | 'pmlb:30' (first 30, small -> large)
        lim = int(spec.split(":")[1]) if ":" in spec else None
        names = [f"pmlb:{n}" for n in PMLB_CURATED]
        return names[:lim] if lim else names
    if spec.startswith("cc18"):
        lim = int(spec.split(":")[1]) if ":" in spec else None
        return list_cc18(limit=lim, **kw)
    if spec == "labs":
        return list(NAMED_OPENML)
    return [s.strip() for s in spec.split(",") if s.strip()]
