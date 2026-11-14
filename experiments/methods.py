"""All compared methods behind one interface: fit_predict(...) -> dict(pred, proba, classes, fit_time, extra)."""
from __future__ import annotations

import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from scout.orchestrator import ScoutAutoML
from scout.preprocess import build_preprocessor, plan_preprocessing
from scout.profile import profile_dataframe

warnings.filterwarnings("ignore")

LOCAL_METHODS = ["rf_default", "hgb_default", "logreg_default", "scout_no_eda", "scout_no_meta", "scout_eda_meta",
                 "scout_bandit", "scout_bandit_nometa"]
EXTERNAL_METHODS = ["flaml", "h2o"]


def _generic(model, scale, Xtr, ytr, seed):
    prof = profile_dataframe(Xtr, ytr, seed, with_outliers=False)
    plan = plan_preprocessing(prof, "classification", adaptive=False)
    pre = build_preprocessor(plan, scale, "classification", seed)
    return Pipeline(pre.steps + [("m", model)])


def _finish(model, Xte, t0, extra=None):
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte) if hasattr(model, "predict_proba") else None
    return dict(pred=pred, proba=proba, classes=np.asarray(model.classes_), fit_time=time.time() - t0, extra=extra or {})


def fit_predict(method: str, Xtr, ytr, Xte, dataset: str, seed: int, memory=None, budget: int = 60, top_k: int = 3,
                n_iter: int = 8):
    t0 = time.time()
    if method == "rf_default":
        m = _generic(RandomForestClassifier(300, n_jobs=1, random_state=seed), False, Xtr, ytr, seed).fit(Xtr, ytr)
        return _finish(m, Xte, t0)
    if method == "hgb_default":
        m = _generic(HistGradientBoostingClassifier(random_state=seed), False, Xtr, ytr, seed).fit(Xtr, ytr)
        return _finish(m, Xte, t0)
    if method == "logreg_default":
        m = _generic(LogisticRegression(max_iter=2000, random_state=seed), True, Xtr, ytr, seed).fit(Xtr, ytr)
        return _finish(m, Xte, t0)
    if method.startswith("scout"):
        if memory is None and method in ("scout_ens_meta5",):
            raise ValueError("memory required")
        cfg = dict(scout_no_eda=dict(use_eda=False, use_meta=False),        # ablation: no EDA-driven preprocessing, no memory
                   scout_no_meta=dict(use_eda=True, use_meta=False),        # EDA agents only (full catalogue search)
                   scout_eda_meta=dict(use_eda=True, use_meta=True),        # full AGENTIC SCOUT v2
                   scout_meta_only=dict(use_eda=False, use_meta=True),
                   scout_bandit=dict(use_eda=True, use_meta=True, search="bandit"),          # full system + UCB bandit search
                   scout_bandit_nometa=dict(use_eda=True, use_meta=False, search="bandit"),
                   scout_ens=dict(use_eda=True, use_meta=False, search="bandit", ensemble_k=3),
                   scout_ens_meta5=dict(use_eda=True, use_meta=True, search="bandit", ensemble_k=3))[method]
        if method == "scout_ens_meta5":
            top_k = 5
        m = ScoutAutoML(seed=seed, memory=memory, exclude_dataset=dataset, top_k=top_k, n_iter=n_iter, **cfg).fit(Xtr, ytr)
        return _finish(m, Xte, t0, dict(best=m.state_.best, candidates=m.state_.candidates, quality=m.state_.profile["quality_score"]))
    if method == "flaml":
        from flaml import AutoML
        Xa, Xb = _to_category(Xtr, Xte)
        a = AutoML()
        a.fit(Xa, np.asarray(ytr), task="classification", time_budget=budget, metric="macro_f1", seed=seed, verbose=0,
              n_jobs=1, eval_method="cv", n_splits=3)
        pred, proba = a.predict(Xb), a.predict_proba(Xb)
        return dict(pred=np.asarray(pred), proba=proba, classes=np.asarray(a.classes_), fit_time=time.time() - t0,
                    extra=dict(best=str(a.best_estimator)))
    if method == "h2o":
        return _h2o(Xtr, ytr, Xte, seed, budget, t0)
    raise KeyError(method)


def _to_category(Xtr, Xte):
    Xa, Xb = Xtr.copy(), Xte.copy()
    for c in Xa.columns:
        if not pd.api.types.is_numeric_dtype(Xa[c]):
            cats = pd.Categorical(pd.concat([Xa[c], Xb[c]]).astype(str)).categories
            Xa[c] = pd.Categorical(Xa[c].astype(str), categories=cats)
            Xb[c] = pd.Categorical(Xb[c].astype(str), categories=cats)
    return Xa, Xb


_H2O_STARTED = False


def _h2o(Xtr, ytr, Xte, seed, budget, t0):
    import h2o
    from h2o.automl import H2OAutoML
    global _H2O_STARTED
    if not _H2O_STARTED:
        h2o.init(max_mem_size="4G", nthreads=1, verbose=False)
        h2o.no_progress()
        _H2O_STARTED = True
    tr = Xtr.copy()
    tr["__y__"] = np.asarray(ytr).astype(str)
    ftr, fte = h2o.H2OFrame(tr), h2o.H2OFrame(Xte)
    ftr["__y__"] = ftr["__y__"].asfactor()
    aml = H2OAutoML(max_runtime_secs=budget, seed=seed, sort_metric="mean_per_class_error", nfolds=3)
    aml.train(x=list(Xtr.columns), y="__y__", training_frame=ftr)
    pr = aml.leader.predict(fte).as_data_frame()
    _num = lambda s: int(float(str(s).lstrip("p")))          # H2O names probability columns p0, p1, ... in recent versions
    labels = sorted([c for c in pr.columns if c != "predict"], key=_num)
    classes = np.array([_num(s) for s in labels])
    proba = pr[labels].to_numpy()
    pred = pr["predict"].astype(float).astype(int).to_numpy()
    return dict(pred=pred, proba=proba, classes=classes, fit_time=time.time() - t0, extra=dict(best=aml.leader.model_id))
