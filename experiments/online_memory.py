"""Outer feedback loop experiment: does a growing experience memory improve algorithm recommendation?
Datasets arrive in random order; for the i-th dataset the meta-learner only sees the previous i-1 datasets
(their stored 5-fold CV balanced accuracy). Metric: regret of the top-1 recommended algorithm w.r.t. the best algorithm
on that dataset (lower is better), compared with a global-rank baseline and a random pick. No new model training."""
import json, numpy as np, pandas as pd
from scout.memory import ExperienceMemory, MetaLearner
mem = ExperienceMemory.load("results/memory.json"); R = mem.records; n = len(R)
rng = np.random.default_rng(0); rows = []
for rep in range(200):
    order = rng.permutation(n); seen = ExperienceMemory()
    for i, j in enumerate(order):
        r = R[j]
        if i >= 3:
            best = max(r["scores"].values())
            ml = MetaLearner(seen); rec = ml.recommend(r["meta"]); glob = ml.global_order()
            ok = lambda lst: next(a for a in lst if a in r["scores"])
            rnd = rng.choice(list(r["scores"]))
            rows.append(dict(rep=rep, mem=i, meta=best - r["scores"][ok(rec)], glob=best - r["scores"][ok(glob)], rand=best - r["scores"][rnd]))
        seen.add(r["name"], r["meta"], r["scores"], r["times"])
d = pd.DataFrame(rows); d["bin"] = pd.cut(d.mem, [2, 8, 14, 20, 26, 30], labels=["3-8", "9-14", "15-20", "21-26", "27-29"])
t = d.groupby("bin", observed=True)[["meta", "glob", "rand"]].mean().round(4); t["n_obs"] = d.groupby("bin", observed=True).size()
t.to_csv("results/online_memory.csv"); print(t); print("overall", d[["meta", "glob", "rand"]].mean().round(4).to_dict())
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=200)
for c, l, col in [("rand", "random pick", "#9aa3ad"), ("glob", "global rank", "#E9A23B"), ("meta", "kNN meta-learner", "#6C5B9E")]:
    ax.plot(t.index.astype(str), t[c], marker="o", label=l, color=col, lw=2)
ax.set_xlabel("datasets already in memory"); ax.set_ylabel("regret of top-1 (balanced acc.)"); ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False); plt.tight_layout(); plt.savefig("results/fig_online_memory.png")
