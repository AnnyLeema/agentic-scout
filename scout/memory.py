"""Experience memory + meta-learning (warm-start algorithm recommendation).

Memory = list of past datasets: {name, meta-features, per-algorithm CV score & time}. The MetaLearner is a
k-nearest-neighbour recommender in standardised meta-feature space (cf. Brazdil et al.; Feurer et al. 2015).
Evaluation is always leave-one-dataset-out: the dataset being solved is excluded from memory.
"""
from __future__ import annotations

import json
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .profile import profile_dataframe
from .preprocess import FrameCleaner

META_KEYS = ["log_n", "log_d", "d_over_n", "n_classes", "imbalance", "class_entropy", "frac_numeric",
             "frac_categorical", "missing_frac", "mean_abs_skew", "max_abs_skew", "mean_kurtosis",
             "mean_abs_corr", "pca95_frac", "lm_stump", "lm_nb", "lm_1nn"]


def meta_features(X: pd.DataFrame, y: pd.Series, seed: int = 0, profile: Optional[dict] = None) -> Dict[str, float]:
    prof = profile or profile_dataframe(X, y, seed, with_outliers=False)
    n, d = X.shape
    types = prof["types"]
    tg = prof["target"]
    p = np.array(list(tg["counts"].values()), dtype=float)
    p = p / p.sum()
    sk = np.abs(list(prof["skew"].values())) if prof["skew"] else np.array([0.0])
    ku = list(prof["kurtosis"].values()) or [0.0]
    num = types["numeric"]
    mac, pca_frac = 0.0, 1.0
    M = None
    if len(num) >= 2:
        M = X[num].apply(pd.to_numeric, errors="coerce")
        C = M.corr().abs().to_numpy()
        iu = np.triu_indices_from(C, k=1)
        v = C[iu][~np.isnan(C[iu])]
        mac = float(v.mean()) if len(v) else 0.0
        try:
            Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(M.iloc[:3000]))
            pc = PCA(0.95, random_state=seed).fit(Z)
            pca_frac = pc.n_components_ / Z.shape[1]
        except Exception:
            pass
    # landmarkers on a <=2000-row subsample, numeric part only
    lm = dict(lm_stump=np.nan, lm_nb=np.nan, lm_1nn=np.nan)
    if M is not None and len(num) >= 1:
        idx = np.random.RandomState(seed).permutation(len(X))[:2000]
        Mi, yi = M.iloc[idx], y.iloc[idx]
        if yi.value_counts().min() >= 3:
            for key, est in (("lm_stump", DecisionTreeClassifier(max_depth=1, random_state=seed)),
                             ("lm_nb", GaussianNB()), ("lm_1nn", KNeighborsClassifier(1))):
                try:
                    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        lm[key] = float(cross_val_score(pipe, Mi, yi, cv=3, scoring="balanced_accuracy").mean())
                except Exception:
                    pass
    out = dict(log_n=np.log10(n), log_d=np.log10(max(d, 1)), d_over_n=d / n, n_classes=float(tg["n_classes"]),
               imbalance=tg["imbalance_ratio"], class_entropy=float(-(p * np.log(p + 1e-12)).sum()),
               frac_numeric=len(num) / max(d, 1), frac_categorical=len(types["categorical"]) / max(d, 1),
               missing_frac=prof["total_missing_frac"], mean_abs_skew=float(np.mean(sk)),
               max_abs_skew=float(np.max(sk)), mean_kurtosis=float(np.mean(ku)), mean_abs_corr=mac,
               pca95_frac=float(pca_frac), **lm)
    return {k: float(out[k]) for k in META_KEYS}


class ExperienceMemory:
    def __init__(self, records: Optional[List[dict]] = None):
        self.records: List[dict] = records or []

    def add(self, name: str, meta: Dict[str, float], scores: Dict[str, float], times: Dict[str, float]):
        self.records = [r for r in self.records if r["name"] != name]
        self.records.append(dict(name=name, meta=meta, scores=scores, times=times))

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.records, f, indent=1)

    @classmethod
    def load(cls, path: str) -> "ExperienceMemory":
        with open(path) as f:
            return cls(json.load(f))

    def __len__(self):
        return len(self.records)


class MetaLearner:
    """kNN recommender over meta-features; returns algorithms ordered by weighted mean rank."""

    def __init__(self, memory: ExperienceMemory, n_neighbors: int = 5):
        self.memory, self.k = memory, n_neighbors

    def _matrix(self, recs):
        return np.array([[r["meta"].get(k, np.nan) for k in META_KEYS] for r in recs], dtype=float)

    def recommend(self, meta: Dict[str, float], exclude: Optional[str] = None, top: Optional[int] = None):
        recs = [r for r in self.memory.records if r["name"] != exclude]
        if len(recs) < 3:
            return None
        A = self._matrix(recs)
        med = np.nanmedian(A, axis=0)
        A = np.where(np.isnan(A), med, A)
        mu, sd = A.mean(0), A.std(0) + 1e-9
        q = np.array([meta.get(k, np.nan) for k in META_KEYS], dtype=float)
        q = np.where(np.isnan(q), med, q)
        dist = np.sqrt((((A - mu) / sd - (q - mu) / sd) ** 2).sum(1))
        nn = np.argsort(dist)[: min(self.k, len(recs))]
        w = 1.0 / (dist[nn] + 1e-6)
        algos = sorted({a for r in recs for a in r["scores"]})
        rank_sum = dict.fromkeys(algos, 0.0)
        wsum = dict.fromkeys(algos, 0.0)
        for wi, i in zip(w, nn):
            sc = recs[i]["scores"]
            ranks = stats.rankdata([-sc[a] for a in sc if a in sc], method="average")
            for a, rk in zip([a for a in sc], ranks):
                rank_sum[a] += wi * rk
                wsum[a] += wi
        agg = {a: rank_sum[a] / wsum[a] for a in algos if wsum[a] > 0}
        order = sorted(agg, key=agg.get)
        return order[:top] if top else order

    def global_order(self, exclude: Optional[str] = None):
        recs = [r for r in self.memory.records if r["name"] != exclude]
        algos = sorted({a for r in recs for a in r["scores"]})
        mr = {a: np.mean([stats.rankdata([-v for v in r["scores"].values()])[list(r["scores"]).index(a)]
                          for r in recs if a in r["scores"]]) for a in algos}
        return sorted(mr, key=mr.get)
