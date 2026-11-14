"""Candidate algorithm catalogue + small hyper-parameter spaces for the ModelAgent."""
from __future__ import annotations

from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# name -> (factory(seed, class_weight), scale_sensitive, search space)
CLASSIFIERS = {
    "logreg": (lambda s, cw: LogisticRegression(max_iter=2000, class_weight="balanced" if cw else None, random_state=s),
               True, {"C": loguniform(1e-3, 1e2)}),
    "knn": (lambda s, cw: KNeighborsClassifier(), True, {"n_neighbors": randint(1, 30), "weights": ["uniform", "distance"]}),
    "gnb": (lambda s, cw: GaussianNB(), True, {"var_smoothing": loguniform(1e-11, 1e-6)}),
    "dtree": (lambda s, cw: DecisionTreeClassifier(class_weight="balanced" if cw else None, random_state=s),
              False, {"max_depth": randint(2, 20), "min_samples_leaf": randint(1, 20)}),
    "rf": (lambda s, cw: RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample" if cw else None,
                                                n_jobs=1, random_state=s),
           False, {"max_features": ["sqrt", "log2", 0.5], "min_samples_leaf": randint(1, 10)}),
    "extratrees": (lambda s, cw: ExtraTreesClassifier(n_estimators=200, class_weight="balanced" if cw else None,
                                                      n_jobs=1, random_state=s),
                   False, {"max_features": ["sqrt", "log2", 0.5], "min_samples_leaf": randint(1, 10)}),
    "hgb": (lambda s, cw: HistGradientBoostingClassifier(class_weight="balanced" if cw else None, random_state=s),
            False, {"learning_rate": loguniform(0.02, 0.3), "max_leaf_nodes": randint(8, 64),
                    "l2_regularization": loguniform(1e-4, 10)}),
    "svm": (lambda s, cw: SVC(probability=True, class_weight="balanced" if cw else None, random_state=s),
            True, {"C": loguniform(1e-2, 1e2), "gamma": loguniform(1e-4, 1e0)}),
    "mlp": (lambda s, cw: MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, early_stopping=True, random_state=s),
            True, {"alpha": loguniform(1e-5, 1e-1), "hidden_layer_sizes": [(32,), (64,), (128,), (64, 32)]}),
}

REGRESSORS = {
    "ridge": (lambda s, cw: Ridge(), True, {"alpha": loguniform(1e-3, 1e3)}),
    "knn": (lambda s, cw: KNeighborsRegressor(), True, {"n_neighbors": randint(1, 30)}),
    "dtree": (lambda s, cw: DecisionTreeRegressor(random_state=s), False, {"max_depth": randint(2, 20)}),
    "rf": (lambda s, cw: RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=s), False,
           {"max_features": [1.0, 0.5, "sqrt"], "min_samples_leaf": randint(1, 10)}),
    "extratrees": (lambda s, cw: ExtraTreesRegressor(n_estimators=200, n_jobs=1, random_state=s), False,
                   {"max_features": [1.0, 0.5, "sqrt"]}),
    "hgb": (lambda s, cw: HistGradientBoostingRegressor(random_state=s), False,
            {"learning_rate": loguniform(0.02, 0.3), "max_leaf_nodes": randint(8, 64)}),
    "svm": (lambda s, cw: SVR(), True, {"C": loguniform(1e-2, 1e2), "gamma": loguniform(1e-4, 1e0)}),
    "mlp": (lambda s, cw: MLPRegressor(hidden_layer_sizes=(64,), max_iter=300, early_stopping=True, random_state=s),
            True, {"alpha": loguniform(1e-5, 1e-1)}),
}


def catalogue(task: str):
    return CLASSIFIERS if task == "classification" else REGRESSORS


ALGO_NAMES = list(CLASSIFIERS)
