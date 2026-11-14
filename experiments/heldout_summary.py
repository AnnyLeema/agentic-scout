"""Summarises the held-out run (16 unseen PMLB datasets) -> results/table_heldout.csv, results/heldout_summary.json.
Raw rows: results/heldout_*.csv (produced by experiments.run_benchmark, see README)."""
import glob, json
import numpy as np, pandas as pd
from scipy import stats
from experiments.analyze import holm

R = "results"
REF = "scout_bandit_nometa"
LAB = {"scout_bandit_nometa": "AGENTIC SCOUT (EDA + bandit)", "scout_ens": "Scout + top-3 ensemble", "scout_bandit": "Scout + memory + bandit",
       "h2o": "H2O AutoML", "flaml": "FLAML", "hgb_default": "HistGB (generic prep)", "rf_default": "RF (generic prep)", "logreg_default": "LogReg (generic prep)"}

df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{R}/heldout_*.csv"))])
df = df[df["balanced_acc"].notna() & df["method"].isin(LAB)]
ba = df.groupby(["dataset", "method"])["balanced_acc"].mean().unstack()
ba = ba.dropna()
rank = ba.rank(axis=1, ascending=False).mean()
fit = df.groupby("method")["fit_time"].mean()
others = [m for m in ba.columns if m != REF]
ps = []
for m in others:
    d = ba[REF] - ba[m]
    ps.append(stats.wilcoxon(ba[REF], ba[m], zero_method="pratt").pvalue)
adj = holm(ps)
vs = {m: dict(p=float(p), p_holm=float(a), wins=int((ba[REF] > ba[m]).sum()), losses=int((ba[REF] < ba[m]).sum()), ties=int((ba[REF] == ba[m]).sum()))
      for m, p, a in zip(others, ps, adj)}
chi, p = stats.friedmanchisquare(*[ba[m] for m in ba.columns])
S = dict(n_datasets=len(ba), datasets=sorted(d.replace("pmlb:", "") for d in ba.index), ref=REF, vs=vs, friedman=dict(chi2=float(chi), p=float(p)),
         mean_ba=ba.mean().to_dict(), rank=rank.to_dict(), fit=fit.to_dict(), rank_h2o=float(rank["h2o"]), rank_flaml=float(rank["flaml"]))
json.dump(S, open(f"{R}/heldout_summary.json", "w"), indent=1)
t = pd.DataFrame({"Method": [LAB[m] for m in ba.columns], "BA": ba.mean().values, "Rank": rank.values, "Fit s": [fit[m] for m in ba.columns]}).sort_values("Rank")
t.to_csv(f"{R}/table_heldout.csv", index=False)
print(t.to_string(index=False)); print({m: round(v["p_holm"], 4) for m, v in vs.items()}, S["friedman"])
