"""Figure: held-out result (rank bars + per-dataset balanced-accuracy gain of Scout over FLAML and H2O)."""
import glob, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
R = "results"
S = json.load(open(f"{R}/heldout_summary.json"))
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{R}/heldout_*.csv"))])
ba = df.dropna(subset=["balanced_acc"]).groupby(["dataset", "method"])["balanced_acc"].mean().unstack().dropna()
lab = {"scout_bandit_nometa": "AGENTIC SCOUT", "scout_ens": "Scout+ensemble", "scout_bandit": "Scout+memory", "h2o": "H2O", "flaml": "FLAML",
       "hgb_default": "HistGB", "rf_default": "RF", "logreg_default": "LogReg"}
rk = pd.Series(S["rank"]).sort_values()
col = ["#1b6ca8" if m == "scout_bandit_nometa" else "#8fb8d8" if m.startswith("scout") else "#b0b0b0" for m in rk.index]
plt.rcParams.update({"font.size": 7, "font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.5), gridspec_kw={"width_ratios": [1, 1.25]})
a = ax[0]
a.barh([lab[m] for m in rk.index][::-1], rk.values[::-1], color=col[::-1])
for i, v in enumerate(rk.values[::-1]): a.text(v + .08, i, f"{v:.2f}", va="center", fontsize=6.5)
a.set_xlabel("Average rank (lower is better)"); a.set_xlim(0, 7); a.set_title("(a) Rank on 16 unseen datasets", fontsize=7.5)
for s in ("top", "right"): a.spines[s].set_visible(False)
b = ax[1]
order = (ba["scout_bandit_nometa"] - ba["flaml"]).sort_values().index
y = np.arange(len(order))
for m, c, mk in (("flaml", "#d95f02", "o"), ("h2o", "#7570b3", "s")):
    b.scatter((ba["scout_bandit_nometa"] - ba[m]).loc[order], y, c=c, marker=mk, s=14, label=f"vs {lab[m]}")
b.axvline(0, color="k", lw=.7)
b.set_yticks(y); b.set_yticklabels([d.replace("pmlb:", "") for d in order], fontsize=5.5)
b.set_xlabel("Balanced-accuracy gain (right of 0: Scout better)"); b.set_title("(b) Per-dataset difference", fontsize=7.5)
b.legend(fontsize=6, frameon=False, loc="lower right")
for s in ("top", "right"): b.spines[s].set_visible(False)
b.grid(axis="x", lw=.3, alpha=.5)
plt.tight_layout(); plt.savefig(f"{R}/fig_heldout.png", dpi=300); print("ok")
