"""Aggregates raw CSVs into tables, statistical tests, figures and results/summary.json (all paper numbers)."""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

NAMES = {"rf_default": "RF (generic prep)", "hgb_default": "HistGB (generic prep)", "logreg_default": "LogReg (generic prep)",
         "scout_no_eda": "Scout w/o EDA & memory", "scout_no_meta": "Scout (EDA agents only)",
         "scout_eda_meta": "AGENTIC SCOUT v2 (full)", "scout_bandit": "Scout + UCB bandit search", "scout_bandit_nometa": "Scout bandit, no memory", "flaml": "FLAML", "h2o": "H2O AutoML"}
R = "results"


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, idx in enumerate(o):
        run = max(run, (len(p) - i) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def benchmark(S):
    f = f"{R}/benchmark.csv"
    if not os.path.exists(f):
        return
    df = pd.read_csv(f)
    if "error" in df:
        S["benchmark_errors"] = int(df["error"].notna().sum())
        df = df[df["error"].isna()] if df["error"].notna().any() else df
    df = df.dropna(subset=["balanced_acc"])
    S["n_datasets"] = int(df.dataset.nunique())
    S["datasets"] = sorted(df.dataset.unique().tolist())
    for metric in ["balanced_acc", "macro_f1", "roc_auc", "log_loss", "accuracy"]:
        if metric not in df:
            continue
        piv = df.groupby(["dataset", "method"])[metric].mean().unstack()
        piv = piv.dropna(axis=1, how="any") if piv.dropna(axis=1, how="any").shape[1] >= 2 else piv
        piv.to_csv(f"{R}/table_{metric}_per_dataset.csv")
        higher = metric != "log_loss"
        ranks = piv.rank(axis=1, ascending=not higher, method="average")
        S[f"mean_{metric}"] = piv.mean().round(4).to_dict()
        S[f"avg_rank_{metric}"] = ranks.mean().round(3).to_dict()
        if metric == "balanced_acc" and piv.shape[0] >= 5 and piv.shape[1] >= 3:
            stat, p = stats.friedmanchisquare(*[piv[c] for c in piv.columns])
            S["friedman"] = dict(chi2=float(stat), p=float(p), n=int(piv.shape[0]), k=int(piv.shape[1]))
            ref = "scout_eda_meta" if "scout_eda_meta" in piv else piv.columns[0]
            others = [c for c in piv.columns if c != ref]
            ps = []
            for c in others:
                try:
                    ps.append(stats.wilcoxon(piv[ref], piv[c]).pvalue)
                except ValueError:
                    ps.append(1.0)
            adj = holm(ps)
            S["wilcoxon_holm_vs_ref"] = {c: dict(p_adj=float(a), wins=int((piv[ref] > piv[c]).sum()),
                                                  ties=int((piv[ref] == piv[c]).sum()), losses=int((piv[ref] < piv[c]).sum()))
                                         for c, a in zip(others, adj)}
            S["reference_method"] = ref
    # time & majority-baseline context
    S["mean_fit_time_s"] = df.groupby("method")["fit_time"].mean().round(2).to_dict()
    maj = df.groupby("dataset")["majority_acc"].mean()
    acc = df.groupby(["dataset", "method"])["accuracy"].mean().unstack()
    S["n_dataset_method_pairs_below_majority_accuracy"] = int(sum((acc.loc[d] < maj[d] - 1e-9).sum() for d in acc.index))
    # chosen algorithms by Scout
    sc = df[df.method == "scout_eda_meta"]
    if len(sc):
        S["scout_selected_counts"] = pd.Series([json.loads(e).get("best") for e in sc.extra.dropna()]).value_counts().to_dict()
    # figure: average ranks
    if "avg_rank_balanced_acc" in S:
        r = pd.Series(S["avg_rank_balanced_acc"]).sort_values()
        fig, ax = plt.subplots(figsize=(4.6, 2.6))
        ax.barh([NAMES.get(k, k) for k in r.index][::-1], r.values[::-1], color="#2563EB")
        ax.set_xlabel("Average rank (lower is better)", fontsize=8)
        for i, v in enumerate(r.values[::-1]):
            ax.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{R}/fig_avg_rank.png", dpi=200)
        plt.close(fig)
    # LaTeX/markdown main table
    cols = [c for c in ["balanced_acc", "macro_f1", "roc_auc", "log_loss"] if f"mean_{c}" in S]
    t = pd.DataFrame({c: S[f"mean_{c}"] for c in cols})
    t["avg_rank(BA)"] = pd.Series(S["avg_rank_balanced_acc"])
    t["fit_time_s"] = pd.Series(S["mean_fit_time_s"])
    t.index = [NAMES.get(i, i) for i in t.index]
    t.round(4).to_csv(f"{R}/table_main.csv")


def meta(S):
    f = f"{R}/meta_eval.csv"
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    S["meta_eval"] = d.groupby("method")[["spearman", "regret_top1", "regret_top3", "hit_top3", "time_ratio"]].mean().round(4).to_dict("index")
    S["meta_n_datasets"] = int(d.dataset.nunique())
    if "meta_knn" in set(d.method) and d.dataset.nunique() >= 6:
        piv = d.pivot(index="dataset", columns="method", values="regret_top3")
        for b in ("global_rank", "random", "default_rf"):
            if b in piv:
                try:
                    S.setdefault("meta_wilcoxon_regret_top3_vs", {})[b] = float(stats.wilcoxon(piv["meta_knn"], piv[b]).pvalue)
                except ValueError:
                    pass


def continual(S):
    f = f"{R}/continual.csv"
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    g = d.groupby(["dataset", "learner", "method", "mem"])[["ACC", "BWT", "Forgetting"]].agg(["mean", "std"]).round(4)
    g.to_csv(f"{R}/table_continual.csv")
    S["continual"] = {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": {m: float(g.loc[k, (m, 'mean')]) for m in ("ACC", "BWT", "Forgetting")} for k in g.index}
    # paired tests herding vs reservoir/naive on forgetting
    tests = {}
    for (ds, lr, M), sub in d[d.method.isin(["herding", "reservoir"])].groupby(["dataset", "learner", "mem"]):
        a = sub[sub.method == "herding"].sort_values("seed").Forgetting.values
        b = sub[sub.method == "reservoir"].sort_values("seed").Forgetting.values
        if len(a) == len(b) and len(a) >= 5:
            try:
                tests[f"{ds}|{lr}|{M}"] = float(stats.wilcoxon(a, b).pvalue)
            except ValueError:
                pass
    S["continual_herding_vs_reservoir_forgetting_p"] = tests
    # figure: forgetting by method for the largest memory
    big = d[(d.mem.isin([0, d.mem.max()]))]
    piv = big.groupby(["learner", "method"])["Forgetting"].mean().unstack()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    pref = [c for c in ["naive", "reservoir", "herding", "joint"] if c in piv]
    rest = [c for c in piv.columns if c not in pref]
    piv[pref + rest].plot.bar(ax=ax, rot=0)
    ax.legend(fontsize=6)
    ax.set_ylabel("Average forgetting (lower is better)")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(f"{R}/fig_forgetting.png", dpi=200)
    plt.close(fig)
    # figure: accuracy matrix heatmaps (seed 0, first dataset/learner, largest memory)
    first = d.iloc[0]
    fig, axs = plt.subplots(1, 3, figsize=(4.8, 1.9))
    for ax, m in zip(axs, ["naive", "reservoir", "herding"]):
        sub = d[(d.dataset == first.dataset) & (d.learner == first.learner) & (d.method == m) & (d.seed == 0)]
        if m != "naive":
            sub = sub[sub.mem == sub.mem.max()]
        if len(sub):
            Rm = np.array(json.loads(sub.iloc[0].R))
            im = ax.imshow(Rm, vmin=0, vmax=1, cmap="viridis")
            ax.set_title(m, fontsize=7)
            ax.tick_params(labelsize=6)
    fig.colorbar(im, ax=axs, fraction=0.02)
    fig.savefig(f"{R}/fig_cl_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    S = {}
    benchmark(S)
    meta(S)
    continual(S)
    json.dump(S, open(f"{R}/summary.json", "w"), indent=1, default=str)
    print(json.dumps(S, indent=1, default=str)[:3500])
