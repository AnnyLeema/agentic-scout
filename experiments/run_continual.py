"""RQ3: class-incremental continual learning, forgetting measurement. Parallel over (dataset, learner, seed)."""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from scout.continual import make_class_incremental, run_stream
from scout.datasets import load_dataset, resolve

warnings.filterwarnings("ignore")
METHODS = ["naive", "reservoir", "herding", "joint", "ewc", "herding_ewc"]


def one(dataset, learner, seed, n_tasks, mem_sizes, epochs, max_rows, lambdas=(100.0, 1000.0, 10000.0)):
    X, y = load_dataset(dataset, max_rows=max_rows, seed=0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)
    y = y.to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    tasks = make_class_incremental(ytr, n_tasks, seed)
    rows = []
    for m in METHODS:
        if m in ("ewc", "herding_ewc") and learner != "softmax":
            continue
        for M in (mem_sizes if m in ("reservoir", "herding", "herding_ewc") else [0]):
            for lam in (lambdas if m in ("ewc", "herding_ewc") else [0.0]):
                r = run_stream(Xtr, ytr, Xte, yte, tasks, m, learner, M, epochs, seed, lam)
                label = f"{m}_l{int(lam)}" if lam else m           # lambda is reported, never selected on test data
                rows.append(dict(dataset=dataset, learner=learner, method=label, mem=M, seed=seed, n_tasks=n_tasks,
                                 ACC=r["ACC"], BWT=r["BWT"], Forgetting=r["Forgetting"],
                                 R=json.dumps(np.round(r["R"], 4).tolist())))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="digits")
    ap.add_argument("--learners", default="sgd,mlp,softmax")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--mem", default="100,300")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-rows", type=int, default=10000)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--out", default="results/continual.csv")
    a = ap.parse_args()
    mems = [int(x) for x in a.mem.split(",")]
    jobs = [(d, l, s) for d in resolve(a.datasets) for l in a.learners.split(",") for s in range(a.seeds)]
    res = Parallel(n_jobs=a.n_jobs)(delayed(one)(d, l, s, a.tasks, mems, a.epochs, a.max_rows) for d, l, s in jobs)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame([r for rows in res for r in rows]).to_csv(a.out, index=False)
    print("done ->", a.out)
