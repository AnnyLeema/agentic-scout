"""RQ1/RQ4: Scout vs baselines + ablations, repeated stratified k-fold, parallel over (dataset, seed, fold).

python -m experiments.run_benchmark --datasets builtin --seeds 3 --folds 5 --n-jobs 4 \
       --methods rf_default,hgb_default,logreg_default,scout_no_eda,scout_no_meta,scout_eda_meta,flaml
(run `--methods h2o --n-jobs 1` separately: one JVM per process.)
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from scout.datasets import load_dataset, resolve
from scout.evaluate import classification_metrics, majority_baseline
from scout.memory import ExperienceMemory
from experiments.methods import fit_predict

warnings.filterwarnings("ignore")


COLS = ['dataset','seed','fold','method','fit_time','majority_acc','n_train','n_features','n_classes','extra',
        'accuracy','balanced_acc','macro_f1','roc_auc','pr_auc','brier','log_loss','error']


def one_task(dataset, seed, fold, methods, memory_path, max_rows, budget, top_k, n_folds=5):
    X, y = load_dataset(dataset, max_rows=max_rows, seed=0)
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    tr, te = list(skf.split(X, y))[fold]
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    mem = ExperienceMemory.load(memory_path) if memory_path and os.path.exists(memory_path) else None
    base = majority_baseline(ytr, yte)
    rows = []
    for m in methods:
        if m in ("scout_eda_meta", "scout_bandit", "scout_ens_meta5") and mem is None:
            continue
        try:
            r = fit_predict(m, Xtr, ytr, Xte, dataset, seed, memory=mem, budget=budget, top_k=top_k)
            met = classification_metrics(yte, r["pred"], r["proba"], r["classes"])
            rows.append(dict(dataset=dataset, seed=seed, fold=fold, method=m, fit_time=r["fit_time"],
                             majority_acc=base["acc"], n_train=len(Xtr), n_features=X.shape[1],
                             n_classes=int(y.nunique()), extra=json.dumps(r["extra"], default=str), **met))
        except Exception as e:
            rows.append(dict(dataset=dataset, seed=seed, fold=fold, method=m, error=str(e)[:200]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="builtin")
    ap.add_argument("--methods", default="rf_default,hgb_default,logreg_default,scout_no_eda,scout_no_meta,scout_eda_meta")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--max-rows", type=int, default=5000)
    ap.add_argument("--budget", type=int, default=60, help="seconds for flaml/h2o")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--memory", default="results/memory.json")
    ap.add_argument("--out", default="results/benchmark.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    methods = [m for m in a.methods.split(",") if m]
    datasets = resolve(a.datasets)
    done = set()
    if os.path.exists(a.out):                                   # resume support
        old = pd.read_csv(a.out)
        done = set(zip(old.dataset, old.seed, old.fold, old.method))
    jobs = []
    for d in datasets:
        for s in range(a.seeds):
            for f in range(a.folds):
                todo = [m for m in methods if (d, s, f, m) not in done]
                if todo:
                    jobs.append((d, s, f, todo))
    print(f"{len(jobs)} work units on {len(datasets)} datasets, n_jobs={a.n_jobs}", flush=True)
    t0 = time.time()
    chunks = [jobs[i:i + max(a.n_jobs * 2, 1)] for i in range(0, len(jobs), max(a.n_jobs * 2, 1))]
    for ch in chunks:
        res = Parallel(n_jobs=a.n_jobs)(delayed(one_task)(d, s, f, ms, a.memory, a.max_rows, a.budget, a.top_k, a.folds) for d, s, f, ms in ch)
        new = pd.DataFrame([r for rows in res for r in rows]).reindex(columns=COLS)
        header = not os.path.exists(a.out)
        new.to_csv(a.out, mode="a", header=header, index=False)
        print(f"  saved chunk; elapsed {time.time() - t0:.0f}s", flush=True)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
