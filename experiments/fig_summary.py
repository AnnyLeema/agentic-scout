"""Summary figure: accuracy distributions, accuracy-vs-time, forgetting, online-memory regret."""
import glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
R = "results"
plt.rcParams.update({"font.size": 6.5, "font.family": "serif"})
BLUE, GREY, OR = "#1b6ca8", "#9a9a9a", "#d95f02"
fig, ax = plt.subplots(2, 2, figsize=(6.2, 4.6))
# (a) dev-set balanced accuracy distribution
d = pd.read_csv(f"{R}/benchmark.csv").dropna(subset=["balanced_acc"])
ba = d.groupby(["dataset", "method"])["balanced_acc"].mean().unstack()
keep = [("scout_bandit_nometa", "Scout (bandit)"), ("scout_eda_meta", "Scout (full)"), ("h2o", "H2O"), ("flaml", "FLAML"), ("hgb_default", "HistGB"), ("rf_default", "RF"), ("logreg_default", "LogReg")]
a = ax[0, 0]
bp = a.boxplot([ba[m].dropna() for m, _ in keep], orientation='horizontal', patch_artist=True, widths=.6, medianprops=dict(color="k"), flierprops=dict(markersize=2))
for p, (m, _) in zip(bp["boxes"], keep): p.set_facecolor(BLUE if m.startswith("scout") else GREY); p.set_alpha(.85)
a.set_yticklabels([l for _, l in keep]); a.invert_yaxis(); a.set_xlabel("Balanced accuracy"); a.set_title("(a) Spread over 30 development datasets", fontsize=7)
# (b) accuracy vs fit time, dev + held-out
h = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/heldout_*.csv")]).dropna(subset=["balanced_acc"])
b = ax[0, 1]
for df, mk, lab in ((d, "o", "30 dev"), (h, "^", "16 held-out")):
    g = df.groupby("method").agg(ba=("balanced_acc", "mean"), t=("fit_time", "mean"))
    for m, nm in keep:
        if m in g.index:
            b.scatter(g.loc[m, "t"], g.loc[m, "ba"], marker=mk, s=22, c=BLUE if m.startswith("scout") else GREY, edgecolor="k", lw=.4)
            if mk == "^": b.annotate(nm, (g.loc[m, "t"], g.loc[m, "ba"]), fontsize=5, xytext=(2, -7) if m == "h2o" else (2, 2), textcoords="offset points")
b.set_xscale("symlog", linthresh=1); b.set_xlabel("Mean fit time per task (s, log scale)"); b.set_ylabel("Mean balanced accuracy")
b.scatter([], [], marker="o", c="w", edgecolor="k", label="30 dev"); b.scatter([], [], marker="^", c="w", edgecolor="k", label="16 held-out")
b.legend(fontsize=5.5, frameon=False, loc="lower right"); b.set_title("(b) Accuracy versus time", fontsize=7)
# (c) forgetting
c = pd.read_csv(f"{R}/continual.csv"); c = c[c.learner == "softmax"]
f = c.groupby(["method", "mem"]).Forgetting.mean()
items = [("Naive", f[("naive", 0)]), ("Replay 100", f[("herding", 100)]), ("Replay 300", f[("herding", 300)]), ("EWC 1000", f[("ewc_l1000", 0)]),
         ("Replay100+EWC1000", f[("herding_ewc_l1000", 100)]), ("Joint (bound)", f[("joint", 0)])]
e = ax[1, 0]; e.bar([i[0] for i in items], [i[1] for i in items], color=[OR] + [BLUE] * 4 + [GREY])
for i, v in enumerate([x[1] for x in items]): e.text(i, v + .006, f"{v:.2f}", ha="center", fontsize=6)
e.set_ylabel("Forgetting (lower is better)"); e.set_title("(c) Continual learning (softmax learner)", fontsize=7); plt.setp(e.get_xticklabels(), rotation=30, ha="right", fontsize=5.8)
# (d) online memory
o = pd.read_csv(f"{R}/online_memory.csv"); g = ax[1, 1]; x = np.arange(len(o))
for col, lab, cc in (("rand", "Random", GREY), ("meta", "kNN", BLUE), ("glob", "Global rank", OR)): g.plot(x, o[col], marker="o", ms=3, c=cc, label=lab)
g.set_xticks(x); g.set_xticklabels(o["bin"]); g.set_xlabel("Datasets already in memory"); g.set_ylabel("Top-1 regret (lower is better)")
g.legend(fontsize=5.5, frameon=False); g.set_title("(d) Feedback loop: memory grows", fontsize=7)
for q in ax.ravel():
    for s in ("top", "right"): q.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(f"{R}/fig_summary.png", dpi=300); print("ok")
