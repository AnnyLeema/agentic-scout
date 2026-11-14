"""RQ2: leave-one-dataset-out evaluation of the meta-learner against global-rank, default and random baselines."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from scout.memory import ExperienceMemory, MetaLearner


def evaluate(mem: ExperienceMemory, k_top=3, k_neighbors=5, n_random=200, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for r in mem.records:
        true = r["scores"]
        algos = list(true)
        best = max(true.values())
        ml = MetaLearner(mem, k_neighbors)
        orders = {
            "meta_knn": ml.recommend(r["meta"], exclude=r["name"]),
            "global_rank": ml.global_order(exclude=r["name"]),
            "default_rf": ["rf"] + [a for a in algos if a != "rf"],
        }
        for method, order in orders.items():
            if order is None:
                continue
            order = [a for a in order if a in true]
            top = order[:k_top]
            pos = {a: i for i, a in enumerate(order)}
            rho = stats.spearmanr([pos[a] for a in algos], [-true[a] for a in algos])[0] if len(algos) > 2 else np.nan
            rows.append(dict(dataset=r["name"], method=method, spearman=rho,
                             regret_top1=best - true[order[0]], regret_top3=best - max(true[a] for a in top),
                             hit_top3=float(max(true, key=true.get) in top),
                             time_ratio=sum(r["times"].get(a, 0) for a in top) / max(sum(r["times"].values()), 1e-9)))
        # random baseline averaged over permutations
        rr = []
        for _ in range(n_random):
            o = list(rng.permutation(algos))
            top = o[:k_top]
            rr.append((stats.spearmanr([o.index(a) for a in algos], [-true[a] for a in algos])[0],
                       best - true[o[0]], best - max(true[a] for a in top), float(max(true, key=true.get) in top)))
        rr = np.nanmean(np.array(rr), axis=0)
        rows.append(dict(dataset=r["name"], method="random", spearman=rr[0], regret_top1=rr[1], regret_top3=rr[2],
                         hit_top3=rr[3], time_ratio=k_top / len(algos)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", default="results/memory.json")
    ap.add_argument("--out", default="results/meta_eval.csv")
    a = ap.parse_args()
    df = evaluate(ExperienceMemory.load(a.memory))
    df.to_csv(a.out, index=False)
    print(df.groupby("method")[["spearman", "regret_top1", "regret_top3", "hit_top3", "time_ratio"]].mean().round(4))
