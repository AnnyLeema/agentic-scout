"""Continual-learning agent: class-incremental streams with forgetting measurement.

Methods
  naive     : fine-tune on the newest task only (lower bound -> catastrophic forgetting)
  reservoir : experience replay with a reservoir-sampled memory of M samples
  herding   : class-balanced exemplar memory selected by herding (iCaRL-style) + replay  <- Scout-CL
  joint     : retrain on all data seen so far (upper bound, not memory-bounded)
  ewc       : Elastic Weight Consolidation (Kirkpatrick et al. 2017) - quadratic penalty on weights important
              to earlier tasks (diagonal empirical Fisher). Needs the `softmax` learner (own numpy implementation).
  herding_ewc : herding replay + EWC combined
Metrics (Lopez-Paz & Ranzato 2017; Chaudhry et al. 2018): ACC, BWT, average forgetting.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def make_class_incremental(y: np.ndarray, n_tasks: int, seed: int) -> List[np.ndarray]:
    classes = np.unique(y)
    rng = np.random.RandomState(seed)
    rng.shuffle(classes)
    return [np.array(sorted(c)) for c in np.array_split(classes, n_tasks)]


def _learner(kind: str, seed: int, ewc_lambda: float = 0.0):
    if kind == "softmax":
        return SoftmaxLearner(ewc_lambda=ewc_lambda, seed=seed)
    if kind == "sgd":
        return SGDClassifier(loss="log_loss", alpha=1e-4, learning_rate="constant", eta0=0.01, random_state=seed)
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(128,), batch_size=64, learning_rate_init=1e-3, random_state=seed)
    raise ValueError(kind)


class SoftmaxLearner:
    """Multinomial logistic regression trained by mini-batch SGD with an optional EWC penalty.

    Loss = CE + l2/2 |W|^2 + (lam/2) * sum_i F_i (theta_i - theta*_i)^2     (theta* = weights after previous task,
    F = diagonal empirical Fisher accumulated over tasks, i.e. 'online EWC').
    """

    def __init__(self, lr=0.05, l2=1e-4, ewc_lambda=0.0, batch=64, seed=0):
        self.lr, self.l2, self.lam, self.batch = lr, l2, ewc_lambda, batch
        self.rng = np.random.RandomState(seed)
        self.W = self.b = self.anchor_W = self.anchor_b = self.FW = self.Fb = None
        self.classes_ = None

    def _init(self, d, classes):
        self.classes_ = np.asarray(classes)
        self.W, self.b = np.zeros((d, len(classes))), np.zeros(len(classes))

    def _proba(self, X):
        Z = X @ self.W + self.b
        Z -= Z.max(1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(1, keepdims=True)

    def _onehot(self, y):
        return (y[:, None] == self.classes_[None, :]).astype(float)

    def partial_fit(self, X, y, classes=None):
        if self.W is None:
            self._init(X.shape[1], classes)
        idx = self.rng.permutation(len(X))
        for s in range(0, len(X), self.batch):
            bi = idx[s:s + self.batch]
            Xb, Yb = X[bi], self._onehot(y[bi])
            G = (self._proba(Xb) - Yb) / len(bi)
            gW, gb = Xb.T @ G + self.l2 * self.W, G.sum(0)
            self.W -= self.lr * gW
            self.b -= self.lr * gb
            if self.lam > 0 and self.FW is not None:
                # implicit (proximal) step for the quadratic EWC penalty: stable for any lambda, unlike an explicit
                # gradient step whose effective rate lr*lam*F can exceed the stability limit.
                kW, kb = self.lr * self.lam * self.FW, self.lr * self.lam * self.Fb
                self.W = (self.W + kW * self.anchor_W) / (1.0 + kW)
                self.b = (self.b + kb * self.anchor_b) / (1.0 + kb)
        return self

    def consolidate(self, X, y):
        """Called after finishing a task: estimate Fisher on that task's data, store anchor weights."""
        P = self._proba(X)
        D = P - self._onehot(y)                                   # d(-log p_y)/d logits, per sample
        FW = np.mean((X[:, :, None] * D[:, None, :]) ** 2, axis=0)  # per-sample gradient squared, averaged
        Fb = np.mean(D ** 2, axis=0)
        self.FW = FW if self.FW is None else self.FW + FW
        self.Fb = Fb if self.Fb is None else self.Fb + Fb
        self.anchor_W, self.anchor_b = self.W.copy(), self.b.copy()

    def predict(self, X):
        return self.classes_[np.argmax(X @ self.W + self.b, axis=1)]


class _Memory:
    def __init__(self, size: int, mode: str, seed: int):
        self.size, self.mode, self.rng = size, mode, np.random.RandomState(seed)
        self.X = np.empty((0, 0))
        self.y = np.empty(0, dtype=int)
        self.seen = 0

    def update(self, X, y, scaler=None):
        if self.size <= 0:
            return
        if self.mode == "reservoir":
            if self.X.size == 0:
                self.X = np.empty((0, X.shape[1]))
            for xi, yi in zip(X, y):
                self.seen += 1
                if len(self.y) < self.size:
                    self.X, self.y = np.vstack([self.X, xi]), np.append(self.y, yi)
                else:
                    j = self.rng.randint(0, self.seen)
                    if j < self.size:
                        self.X[j], self.y[j] = xi, yi
        else:  # herding: keep a class-balanced set, exemplars nearest to the class mean (iterative greedy)
            allX = np.vstack([self.X, X]) if self.X.size else X
            ally = np.concatenate([self.y, y])
            classes = np.unique(ally)
            m = max(1, self.size // len(classes))
            keepX, keepy = [], []
            for c in classes:
                Xc = allX[ally == c]
                Z = scaler.transform(Xc) if scaler is not None else Xc
                mu = Z.mean(0)
                chosen, running = [], np.zeros_like(mu)
                for k in range(1, min(m, len(Z)) + 1):
                    cand = np.linalg.norm(mu - (running + Z) / k, axis=1)
                    cand[chosen] = np.inf
                    i = int(np.argmin(cand))
                    chosen.append(i)
                    running += Z[i]
                keepX.append(Xc[chosen])
                keepy.append(np.full(len(chosen), c))
            self.X, self.y = np.vstack(keepX), np.concatenate(keepy)


def run_stream(Xtr, ytr, Xte, yte, tasks, method: str, learner: str = "sgd", mem_size: int = 200, epochs: int = 10,
               seed: int = 0, ewc_lambda: float = 0.0) -> Dict:
    """Returns accuracy matrix R[i, j] = acc on task j's test data after training through task i
    (prediction over ALL classes -> class-incremental, the hardest scenario of van de Ven & Tolias 2019)."""
    T = len(tasks)
    all_classes = np.unique(ytr)
    scaler = StandardScaler()
    use_ewc = method in ("ewc", "herding_ewc")
    if use_ewc and learner != "softmax":
        raise ValueError("EWC is implemented for the 'softmax' learner only")
    clf = _learner(learner, seed, ewc_lambda if use_ewc else 0.0)
    mem_mode = {"reservoir": "reservoir", "herding": "herding", "herding_ewc": "herding"}.get(method)
    mem = _Memory(mem_size, mem_mode, seed) if mem_mode else None
    rng = np.random.RandomState(seed)
    R = np.full((T, T), np.nan)
    seenX, seeny = [], []
    for t, cl in enumerate(tasks):
        mask = np.isin(ytr, cl)
        Xt, yt = Xtr[mask], ytr[mask]
        scaler.partial_fit(Xt)
        if method == "joint":
            seenX.append(Xt), seeny.append(yt)
            Xfit, yfit = np.vstack(seenX), np.concatenate(seeny)
            clf = _learner(learner, seed)
        elif mem is not None and len(mem.y):
            Xfit, yfit = np.vstack([Xt, mem.X]), np.concatenate([yt, mem.y])
        else:
            Xfit, yfit = Xt, yt
        reps = epochs if method != "joint" else epochs
        for _ in range(reps):
            p = rng.permutation(len(Xfit))
            clf.partial_fit(scaler.transform(Xfit[p]), yfit[p], classes=all_classes)
        if use_ewc:
            clf.consolidate(scaler.transform(Xt), yt)
        if mem is not None:
            mem.update(Xt, yt, scaler)
        for j, cj in enumerate(tasks):
            mj = np.isin(yte, cj)
            R[t, j] = float((clf.predict(scaler.transform(Xte[mj])) == yte[mj]).mean())
    return dict(R=R, **cl_metrics(R))


def cl_metrics(R: np.ndarray) -> Dict[str, float]:
    T = R.shape[0]
    acc = float(np.mean(R[T - 1]))
    bwt = float(np.mean([R[T - 1, j] - R[j, j] for j in range(T - 1)])) if T > 1 else 0.0
    forg = float(np.mean([np.max(R[j:T - 1, j]) - R[T - 1, j] for j in range(T - 1)])) if T > 1 else 0.0
    return dict(ACC=acc, BWT=bwt, Forgetting=forg)
