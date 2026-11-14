"""EDA visualisations from syllabus modules 3-5 (matplotlib only)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def _num(X):
    return X.select_dtypes(include=[np.number])


def histograms(X, max_cols=12):
    N = _num(X).iloc[:, :max_cols]
    n = N.shape[1]
    r, c = (n + 3) // 4, min(4, n)
    fig, axs = plt.subplots(r, c, figsize=(3 * c, 2.2 * r), squeeze=False)
    for ax, col in zip(axs.ravel(), N.columns):
        ax.hist(N[col].dropna(), bins=30, color="#2563EB")
        ax.set_title(str(col)[:22], fontsize=8)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def boxplots(X, max_cols=12):
    N = _num(X).iloc[:, :max_cols]
    Z = (N - N.mean()) / (N.std() + 1e-9)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.boxplot([Z[c].dropna() for c in Z.columns])
    ax.set_xticks(range(1, len(Z.columns) + 1))
    ax.set_xticklabels([str(c)[:10] for c in Z.columns], rotation=60, fontsize=7)
    ax.set_title("Standardised boxplots (IQR outliers)")
    fig.tight_layout()
    return fig


def corr_heatmap(X, max_cols=25):
    C = _num(X).iloc[:, :max_cols].corr()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(C)))
    ax.set_xticklabels([str(c)[:8] for c in C.columns], rotation=90, fontsize=6)
    ax.set_yticks(range(len(C)))
    ax.set_yticklabels([str(c)[:8] for c in C.columns], fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("Correlation heatmap (association, not causation)", fontsize=9)
    fig.tight_layout()
    return fig


def violin_by_target(X, y, max_cols=6):
    N = _num(X).iloc[:, :max_cols]
    classes = sorted(pd.Series(y).unique())[:6]
    fig, axs = plt.subplots(1, N.shape[1], figsize=(2.4 * N.shape[1], 2.8), squeeze=False)
    for ax, c in zip(axs.ravel(), N.columns):
        data = [N.loc[(np.asarray(y) == k), c].dropna() for k in classes]
        data = [d if len(d) > 1 else pd.Series([0, 0]) for d in data]
        ax.violinplot(data, showmedians=True)
        ax.set_title(str(c)[:14], fontsize=8)
        ax.set_xticks(range(1, len(classes) + 1))
        ax.set_xticklabels([str(k) for k in classes], fontsize=7)
    fig.tight_layout()
    return fig


def missingness(X):
    m = X.isna().mean().sort_values(ascending=False)
    m = m[m > 0].head(20)
    fig, ax = plt.subplots(figsize=(6, 2.5))
    if len(m):
        ax.barh(m.index.astype(str)[::-1], m.values[::-1] * 100, color="#F59E0B")
    ax.set_xlabel("% missing")
    fig.tight_layout()
    return fig


def embedding(X, y, method="pca", max_rows=2000, seed=0):
    N = _num(X)
    if N.shape[1] < 2:
        return None
    idx = np.random.RandomState(seed).permutation(len(N))[:max_rows]
    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(N.iloc[idx]))
    E = PCA(2, random_state=seed).fit_transform(Z) if method == "pca" else TSNE(2, random_state=seed, init="pca").fit_transform(Z)
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    sc = ax.scatter(E[:, 0], E[:, 1], c=np.asarray(y)[idx], cmap="tab10", s=8)
    ax.set_title(method.upper() + " projection")
    fig.tight_layout()
    return fig


def parallel_coordinates(X, y, max_cols=8, max_rows=300, seed=0):
    N = _num(X).iloc[:, :max_cols]
    idx = np.random.RandomState(seed).permutation(len(N))[:max_rows]
    Z = (N - N.min()) / (N.max() - N.min() + 1e-9)
    fig, ax = plt.subplots(figsize=(7, 3))
    cmap = plt.get_cmap("tab10")
    yy = np.asarray(y)
    for i in idx:
        ax.plot(range(Z.shape[1]), Z.iloc[i].values, color=cmap(int(yy[i]) % 10), alpha=0.25, lw=0.7)
    ax.set_xticks(range(Z.shape[1]))
    ax.set_xticklabels([str(c)[:8] for c in Z.columns], rotation=45, fontsize=7)
    ax.set_title("Parallel coordinates")
    fig.tight_layout()
    return fig


def seasonal_decomposition(series: pd.Series, period: int):
    from statsmodels.tsa.seasonal import seasonal_decompose
    r = seasonal_decompose(series.dropna(), period=period, model="additive")
    fig = r.plot()
    fig.set_size_inches(6, 5)
    return fig
