"""Multi-agent orchestrator. Every agent is a deterministic component that reads/writes a shared state and
appends to an audit trail; no LLM and no external API are used (reproducible given a seed)."""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline

from .memory import ExperienceMemory, MetaLearner, meta_features
from .models import catalogue
from .preprocess import build_preprocessor, plan_preprocessing
from .profile import profile_dataframe

warnings.filterwarnings("ignore")


@dataclass
class State:
    audit: List[Dict[str, Any]] = field(default_factory=list)
    profile: Optional[dict] = None
    task: Optional[str] = None
    plan: Optional[dict] = None
    meta: Optional[dict] = None
    candidates: Optional[List[str]] = None
    cv_scores: Dict[str, float] = field(default_factory=dict)
    cv_times: Dict[str, float] = field(default_factory=dict)
    best: Optional[str] = None
    search_time: float = 0.0
    bandit_history: Optional[list] = None
    arm_best: Optional[dict] = None

    def log(self, agent: str, decision: str, detail: Any = None):
        self.audit.append(dict(t=time.time(), agent=agent, decision=decision, detail=detail))


class TaskDetectorAgent:
    """Deterministic rules (module 1): target dtype + distinct-value ratio decide the task."""

    def run(self, y: Optional[pd.Series], st: State, forced: Optional[str] = None):
        if forced:
            st.task = forced
            reason = "user-specified"
        elif y is None:
            st.task, reason = "clustering", "no target supplied"
        else:
            ny = y.nunique()
            if pd.api.types.is_numeric_dtype(y) and (ny > 15 and ny / len(y) > 0.05):
                st.task, reason = "regression", f"numeric target with {ny} distinct values"
            else:
                st.task, reason = "classification", f"{ny} distinct target values"
        st.log("TaskDetector", st.task, reason)


class ScoutAutoML(BaseEstimator):
    """sklearn-style estimator wrapping the agent pipeline.

    use_eda   : adaptive, profile-driven preprocessing (False -> fixed generic preprocessing)
    use_meta  : meta-learning warm start from `memory` (False -> evaluate the full catalogue)
    top_k     : number of algorithms proposed by the meta-learner
    """

    def __init__(self, task: Optional[str] = None, use_eda=True, use_meta=True, memory: Optional[ExperienceMemory] = None,
                 exclude_dataset: Optional[str] = None, top_k=3, n_iter=8, inner_cv=3, tune_top=2, seed=0,
                 time_budget_s: Optional[float] = None, algos: Optional[List[str]] = None,
                 search: str = "screen_tune", budget_evals: int = 18, ucb_c: float = 0.1,
                 ensemble_k: int = 1, learn_online: bool = False, dataset_name: Optional[str] = None, memory_path: Optional[str] = None):
        self.task, self.use_eda, self.use_meta, self.memory = task, use_eda, use_meta, memory
        self.exclude_dataset, self.top_k, self.n_iter, self.inner_cv = exclude_dataset, top_k, n_iter, inner_cv
        self.tune_top, self.seed, self.time_budget_s, self.algos = tune_top, seed, time_budget_s, algos
        self.search, self.budget_evals, self.ucb_c = search, budget_evals, ucb_c
        self.ensemble_k = ensemble_k
        self.learn_online, self.dataset_name, self.memory_path = learn_online, dataset_name, memory_path

    # ------------------------------------------------------------------
    def _scoring(self):
        return "balanced_accuracy" if self.state_.task == "classification" else "r2"

    def _bandit(self, pipes, X, y, cv, st):
        """UCB1-style multi-armed bandit over algorithms (reinforcement-learning-flavoured search).

        arm = algorithm; pull = evaluate ONE configuration (default first, then random samples) by inner CV;
        reward = CV score. Index = best score seen by the arm + c*sqrt(2 ln t / n_arm) (max-bandit, Auer et al. 2002).
        Total budget = `budget_evals` CV evaluations, directly comparable to screen-and-tune.
        """
        from sklearn.model_selection import ParameterSampler
        arms = list(pipes)
        B = self.budget_evals
        samples = {}
        for a in arms:
            grid = {f"m__{k}": v for k, v in pipes[a][1].items()}
            try:
                samples[a] = list(ParameterSampler(grid, n_iter=B, random_state=self.seed)) if grid else []
            except Exception:
                samples[a] = []
        n = dict.fromkeys(arms, 0)
        best = dict.fromkeys(arms, -np.inf)
        best_params = {a: {} for a in arms}
        history = []
        for t in range(1, B + 1):
            untried = [a for a in arms if n[a] == 0]
            if untried:
                a, params = untried[0], {}
            else:
                live = [a for a in arms if n[a] - 1 < len(samples[a])]
                if not live:
                    break
                a = max(live, key=lambda k: best[k] + self.ucb_c * np.sqrt(2 * np.log(t) / n[k]))
                params = samples[a][n[a] - 1]
            est = clone(pipes[a][0]).set_params(**params)
            ts = time.time()
            try:
                sc = float(np.nanmean(cross_val_score(est, X, y, cv=cv, scoring=self._scoring(), error_score=np.nan)))
            except Exception:
                sc = float("nan")
            st.cv_times[a] = st.cv_times.get(a, 0.0) + time.time() - ts
            n[a] += 1
            history.append((t, a, sc))
            if np.isfinite(sc) and sc > best[a]:
                best[a], best_params[a] = sc, params
        ok = {a: v for a, v in best.items() if np.isfinite(v)}
        for a, v in ok.items():
            st.cv_scores[a] = float(v)
        winner = max(ok, key=ok.get)
        st.log("Bandit", f"pulls per arm {dict((a, n[a]) for a in arms)}", dict(winner=winner, score=float(ok[winner])))
        st.bandit_history = history
        st.arm_best = {a: (float(v), best_params[a]) for a, v in ok.items()}
        return winner, clone(pipes[a if False else winner][0]).set_params(**best_params[winner]), float(ok[winner])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Run the agent pipeline, then (outer feedback loop) write the experience gained into the memory."""
        self._fit_core(X, y)
        self._write_back(X, y)
        return self

    def _write_back(self, X, y):
        """Outer feedback loop: store this dataset's meta-features and the CV scores observed by the model agent,
        so that the next run's meta-learner can retrieve it. Only classification runs with a memory are stored."""
        st = self.state_
        if not (self.learn_online and self.memory is not None and st.task == "classification" and st.cv_scores):
            return
        if st.meta is None:
            st.meta = meta_features(X, pd.Series(np.asarray(y)), self.seed, st.profile)
        name = self.dataset_name or f"run_{len(self.memory) + 1}"
        scores = {a: v for a, v in st.cv_scores.items() if np.isfinite(v)}
        self.memory.add(name, st.meta, scores, dict(st.cv_times))
        if self.memory_path:
            self.memory.save(self.memory_path)
        st.log("MemoryWriter", f"stored '{name}' in memory", f"memory size={len(self.memory)}")

    def _fit_core(self, X: pd.DataFrame, y: pd.Series):
        t0 = time.time()
        st = self.state_ = State()
        y = pd.Series(np.asarray(y)).reset_index(drop=True)
        X = X.reset_index(drop=True)
        TaskDetectorAgent().run(y, st, self.task)
        task = st.task
        st.profile = profile_dataframe(X, y if task == "classification" else None, self.seed)
        st.log("Profiler", f"quality={st.profile['quality_score']:.1f}",
               dict(missing=st.profile["total_missing_frac"], dups=st.profile["duplicates"],
                    bias=st.profile["bias_flags"], mechanisms=st.profile["missing_mechanism"]))
        st.plan = plan_preprocessing(st.profile, task, adaptive=self.use_eda)
        for r in st.plan["reasons"]:
            st.log("Preprocessor", r)
        cat = catalogue(task)
        names = list(self.algos or cat)
        if task == "classification":
            counts = y.value_counts()
            if counts.min() < 3:
                raise ValueError("fewer than 3 samples in a class; cannot cross-validate")
            if len(X) > 4000 and "svm" in names:
                names.remove("svm")
                st.log("ModelAgent", "excluded SVM", "n > 4000 (probability calibration too slow)")
        # ---- meta-learning agent (memory warm start)
        if self.use_meta and self.memory is not None and task == "classification":
            st.meta = meta_features(X, y, self.seed, st.profile)
            rec = MetaLearner(self.memory).recommend(st.meta, exclude=self.exclude_dataset)
            if rec:
                cand = [a for a in rec if a in names][: self.top_k]
                st.log("MetaLearner", f"warm-start with {cand}", f"memory size={len(self.memory)}")
                names = cand
        st.candidates = names
        # ---- model agent: inner CV over candidates, tune the best few, refit winner
        cv = (StratifiedKFold(self.inner_cv, shuffle=True, random_state=self.seed) if task == "classification"
              else KFold(self.inner_cv, shuffle=True, random_state=self.seed))
        cw = bool(st.plan["class_weight"]) and self.use_eda
        pipes = {}
        for a in names:
            fac, scale, space = cat[a]
            pre = build_preprocessor(st.plan, scale, task, self.seed)
            pipes[a] = (Pipeline(pre.steps + [("m", fac(self.seed, cw))]), space)
        if self.search == "bandit":
            best_name, best_est, best_score = self._bandit(pipes, X, y, cv, st)
            st.best = best_name
            if self.ensemble_k > 1 and st.arm_best and len(st.arm_best) > 1:
                # ensemble agent: soft vote over the k best arms, each with its best configuration found by the bandit
                from sklearn.ensemble import VotingClassifier
                top = sorted(st.arm_best, key=lambda a: -st.arm_best[a][0])[: self.ensemble_k]
                members = [(a, clone(pipes[a][0]).set_params(**st.arm_best[a][1])) for a in top]
                best_est = VotingClassifier(members, voting="soft")
                st.log("EnsembleAgent", f"soft vote over {top}", None)
            self.model_ = best_est.fit(X, y)
            st.search_time = time.time() - t0
            st.log("ModelAgent", f"selected {best_name} (bandit)", dict(inner_cv_score=best_score, seconds=round(st.search_time, 2)))
            self.classes_ = getattr(self.model_, "classes_", None)
            return self
        for a in names:
            ts = time.time()
            try:
                sc = cross_val_score(pipes[a][0], X, y, cv=cv, scoring=self._scoring(), error_score=np.nan)
                st.cv_scores[a] = float(np.nanmean(sc))
            except Exception as e:  # algorithm not applicable
                st.cv_scores[a] = float("nan")
                st.log("ModelAgent", f"{a} failed", str(e)[:120])
            st.cv_times[a] = time.time() - ts
            if self.time_budget_s and time.time() - t0 > self.time_budget_s:
                st.log("ModelAgent", "time budget reached", f"after {a}")
                break
        ok = {a: s for a, s in st.cv_scores.items() if np.isfinite(s)}
        ranked = sorted(ok, key=ok.get, reverse=True)
        best_name, best_est, best_score = ranked[0], pipes[ranked[0]][0], ok[ranked[0]]
        for a in ranked[: self.tune_top]:
            pipe, space = pipes[a]
            if not space or self.n_iter <= 0:
                continue
            grid = {f"m__{k}": v for k, v in space.items()}
            try:
                rs = RandomizedSearchCV(pipe, grid, n_iter=self.n_iter, cv=cv, scoring=self._scoring(),
                                        random_state=self.seed, n_jobs=1, error_score=np.nan)
                rs.fit(X, y)
                st.log("ModelAgent", f"tuned {a}", dict(score=float(rs.best_score_), params={k: str(v) for k, v in rs.best_params_.items()}))
                if rs.best_score_ > best_score:
                    best_name, best_est, best_score = a, rs.best_estimator_, float(rs.best_score_)
            except Exception as e:
                st.log("ModelAgent", f"tuning {a} failed", str(e)[:120])
        st.best = best_name
        self.model_ = best_est.fit(X, y)
        st.search_time = time.time() - t0
        st.log("ModelAgent", f"selected {best_name}", dict(inner_cv_score=float(best_score), seconds=round(st.search_time, 2)))
        self.classes_ = getattr(self.model_, "classes_", None)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    @property
    def audit_trail(self):
        return self.state_.audit
