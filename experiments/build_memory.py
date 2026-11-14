"""Builds the experience memory: meta-features + 5-fold CV score/time of every candidate algorithm per dataset."""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from scout.datasets import load_dataset, resolve
from scout.memory import ExperienceMemory, meta_features
from scout.models import CLASSIFIERS
from scout.preprocess import build_preprocessor, plan_preprocessing
from scout.profile import profile_dataframe

warnings.filterwarnings("ignore")


def one(name, max_rows, seed=0):
    X, y = load_dataset(name, max_rows=max_rows, seed=0)
    prof = profile_dataframe(X, y, seed, with_outliers=False)
    plan = plan_preprocessing(prof, "classification", adaptive=True)
    mf = meta_features(X, y, seed, prof)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    scores, times = {}, {}
    for a, (fac, scale, _) in CLASSIFIERS.items():
        if a == "svm" and len(X) > 4000:
            continue
        pipe = Pipeline(build_preprocessor(plan, scale, "classification", seed).steps + [("m", fac(seed, bool(plan["class_weight"])))])
        t = time.time()
        try:
            scores[a] = float(np.nanmean(cross_val_score(pipe, X, y, cv=cv, scoring="balanced_accuracy", error_score=np.nan)))
            times[a] = time.time() - t
        except Exception:
            pass
    return name, mf, scores, times


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="builtin")
    ap.add_argument("--max-rows", type=int, default=5000)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--out", default="results/memory.json")
    a = ap.parse_args()
    res = Parallel(n_jobs=a.n_jobs)(delayed(one)(d, a.max_rows) for d in resolve(a.datasets))
    mem = ExperienceMemory()
    for name, mf, sc, ti in res:
        if sc:
            mem.add(name, mf, sc, ti)
    mem.save(a.out)
    print(f"memory with {len(mem)} datasets -> {a.out}")
