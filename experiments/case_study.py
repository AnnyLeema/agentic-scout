"""Case study = the ORIGINAL v1 retail-returns experiment, re-run properly with v2 (fixes the v1 reporting problems).

python -m experiments.case_study --csv "india retail sales dataset.csv" --target Returned
Reports: majority baseline on the SAME split, full confusion matrix, per-class P/R/F1, PR-AUC, balanced accuracy,
ROC-AUC, permutation importance, and the rule-based recommendations' caveat. Saves results/case_study.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from experiments.methods import fit_predict
from scout.evaluate import classification_metrics, confusion, explain, majority_baseline
from scout.memory import ExperienceMemory
from scout.orchestrator import ScoutAutoML
from scout.profile import profile_dataframe

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", default="Returned")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--memory", default="results/memory.json")
    ap.add_argument("--out", default="results/case_study.json")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    y = pd.Series(pd.factorize(df[a.target].astype(str).str.strip().str.lower().map(lambda s: s))[0])
    classes_txt = list(pd.factorize(df[a.target].astype(str).str.strip().str.lower())[1])
    X = df.drop(columns=[a.target])
    mem = ExperienceMemory.load(a.memory) if __import__("os").path.exists(a.memory) else None
    out = dict(n=len(df), classes=classes_txt, class_counts=y.value_counts().to_dict(), runs=[])
    out["profile"] = {k: v for k, v in profile_dataframe(X, y).items() if k in
                      ("n_rows", "n_cols", "quality_score", "total_missing_frac", "duplicates", "high_cardinality",
                       "bias_flags", "missing_mechanism", "quality_penalties")}
    for seed in range(a.seeds):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        base = majority_baseline(ytr, yte)
        for m in ["rf_default", "scout_no_meta", "scout_eda_meta"]:
            if m == "scout_eda_meta" and mem is None:
                continue
            r = fit_predict(m, Xtr, ytr, Xte, "__case__", seed, memory=mem)
            met = classification_metrics(yte, r["pred"], r["proba"], r["classes"])
            run = dict(seed=seed, method=m, majority=base, metrics=met,
                       confusion=confusion(yte, r["pred"], r["classes"]).tolist(),
                       report=classification_report(yte, r["pred"], output_dict=True, zero_division=0))
            if m == "scout_no_meta" and seed == 0:
                sc = ScoutAutoML(use_meta=False, seed=seed).fit(Xtr, ytr)
                run["top_features"] = explain(sc.model_, Xte, yte).to_dict("records")
                run["audit"] = [dict(agent=x["agent"], decision=x["decision"]) for x in sc.audit_trail]
            out["runs"].append(run)
    json.dump(out, open(a.out, "w"), indent=1, default=str)
    print("saved", a.out)
