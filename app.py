"""AGENTIC SCOUT v2 - Streamlit interface.   Run:  streamlit run app.py"""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from scout import viz
from scout.continual import make_class_incremental, run_stream
from scout.datasets import load_dataset
from scout.evaluate import classification_metrics, confusion, explain, majority_baseline, regression_metrics
from scout.memory import ExperienceMemory
from scout.orchestrator import ScoutAutoML

st.set_page_config(page_title="AGENTIC SCOUT v2", layout="wide")
st.title("AGENTIC SCOUT v2")
st.caption("Deterministic multi-agent AutoML: EDA agents -> meta-learning memory -> model search -> multi-metric "
           "evaluation -> explanation -> continual learning. No LLM / external API is used.")

MEM_PATH = "results/memory.json"
memory = ExperienceMemory.load(MEM_PATH) if os.path.exists(MEM_PATH) else None

with st.sidebar:
    st.header("1. Data")
    src = st.radio("Source", ["Built-in benchmark", "Upload CSV/Excel"])
    if src == "Built-in benchmark":
        name = st.selectbox("Dataset", ["breast_cancer", "wine", "iris", "digits"])
        X, y = load_dataset(name)
    else:
        up = st.file_uploader("File", type=["csv", "xlsx"])
        if up is None:
            st.stop()
        df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
        target = st.selectbox("Target column", df.columns[::-1])
        y, X = df[target], df.drop(columns=[target])
        name = up.name
        keep = y.notna()
        X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
        if y.dtype == object or y.nunique() <= 15:
            y = pd.Series(pd.factorize(y.astype(str))[0])
    st.header("2. Agents")
    use_eda = st.toggle("EDA-driven preprocessing", True)
    use_meta = st.toggle("Meta-learning warm start", memory is not None, disabled=memory is None)
    if memory is None:
        st.caption("No experience memory found. Run `python -m experiments.build_memory` first.")
    search = st.selectbox("Model-search strategy", ["screen_tune", "bandit"],
                          help="bandit = UCB multi-armed bandit over algorithms (reward = CV score)")
    n_iter = st.slider("Random-search iterations (screen_tune)", 0, 25, 6)
    test_size = st.slider("Test fraction", 0.1, 0.5, 0.25)

tabs = st.tabs(["Overview & quality", "EDA", "Preprocessing plan", "AutoML run", "Evaluation", "Explainability",
                "Experience memory", "Continual learning"])

# ---- Overview / quality (modules 1-2)
from scout.profile import profile_dataframe
prof = profile_dataframe(X, y, with_outliers=True)
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Rows x cols", f"{prof['n_rows']} x {prof['n_cols']}")
    c[1].metric("Data-quality score", f"{prof['quality_score']:.1f}/100")
    c[2].metric("Missing", f"{prof['total_missing_frac']:.2%}")
    c[3].metric("Duplicates", prof["duplicates"])
    st.write("**Quality penalty breakdown** (transparent formula)")
    st.dataframe(pd.Series(prof["quality_penalties"], name="penalty").round(2))
    if prof["target"]:
        st.write("**Target distribution**", prof["target"]["counts"])
    for b in prof["bias_flags"]:
        st.warning(b)
    if prof["missing_mechanism"]:
        st.write("**Missingness diagnostics** (MNAR cannot be tested from observed data)")
        st.dataframe(pd.DataFrame({"missing_frac": prof["missing"], "mechanism": prof["missing_mechanism"]}).dropna())
    st.write(f"Isolation-Forest outlier fraction: {prof['outlier_isolation_forest']:.2%}; "
             f"highly-correlated pairs (|r|>0.9): {prof['n_corr_pairs_gt_09']}")

# ---- EDA (module 3-5)
with tabs[1]:
    st.write("Descriptive statistics"); st.dataframe(X.describe(include="all").T)
    a, b = st.columns(2)
    a.pyplot(viz.histograms(X)); b.pyplot(viz.boxplots(X))
    a.pyplot(viz.corr_heatmap(X)); b.pyplot(viz.violin_by_target(X, y))
    if X.isna().any().any():
        a.pyplot(viz.missingness(X))
    p = viz.embedding(X, y, "pca")
    if p is not None:
        b.pyplot(p)
        if st.checkbox("t-SNE (slow)"):
            st.pyplot(viz.embedding(X, y, "tsne"))
        st.pyplot(viz.parallel_coordinates(X, y))

# ---- run
if "model" not in st.session_state or st.session_state.get("key") != (name, use_eda, use_meta, n_iter, test_size, search):
    st.session_state.pop("model", None)
with tabs[3]:
    if st.button("Run AGENTIC SCOUT", type="primary"):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y if y.nunique() < 50 else None, random_state=0)
        with st.spinner("Agents working..."):
            m = ScoutAutoML(use_eda=use_eda, use_meta=use_meta, memory=memory, exclude_dataset=name, n_iter=n_iter, search=search).fit(Xtr, ytr)
        st.session_state.update(model=m, split=(Xtr, Xte, ytr, yte), key=(name, use_eda, use_meta, n_iter, test_size, search))
    m = st.session_state.get("model")
    if m:
        stt = m.state_
        st.success(f"Task: {stt.task} | Selected: {stt.best} | Search time: {stt.search_time:.1f}s")
        st.write("Candidates evaluated (inner-CV balanced accuracy / R2):")
        st.dataframe(pd.Series(stt.cv_scores, name="inner_cv").sort_values(ascending=False))
        st.write("**Audit trail** (every agent decision and its reason)")
        st.dataframe(pd.DataFrame([{k: (json.dumps(v, default=str)[:160] if k == "detail" else v) for k, v in a.items() if k != "t"}
                                   for a in m.audit_trail]))
    else:
        st.info("Press the button to run the pipeline.")
with tabs[2]:
    from scout.preprocess import plan_preprocessing
    plan = plan_preprocessing(prof, "classification", adaptive=use_eda)
    st.write("Each rule that fired, with its reason:")
    for r in plan["reasons"] or ["(no adaptive rule fired)"]:
        st.write("-", r)

m = st.session_state.get("model")
with tabs[4]:
    if m:
        Xtr, Xte, ytr, yte = st.session_state["split"]
        pred, proba = m.predict(Xte), (m.predict_proba(Xte) if m.state_.task == "classification" else None)
        if m.state_.task == "classification":
            met = classification_metrics(yte, pred, proba, m.classes_)
            base = majority_baseline(ytr, yte)
            st.dataframe(pd.Series(met, name="held-out").round(4))
            st.info(f"Majority-class baseline on the same test split: accuracy {base['acc']:.3f}, balanced accuracy {base['bal_acc']:.3f}. "
                    "A model must beat these to be useful.")
            cm = confusion(yte, pred, m.classes_)
            st.write("Confusion matrix (rows = true)"); st.dataframe(pd.DataFrame(cm, index=m.classes_, columns=m.classes_))
        else:
            st.dataframe(pd.Series(regression_metrics(yte, pred)).round(4))
    else:
        st.info("Run the pipeline first.")
with tabs[5]:
    if m:
        Xtr, Xte, ytr, yte = st.session_state["split"]
        imp = explain(m.model_, Xte, yte, m.state_.task)
        st.bar_chart(imp.set_index("feature")["importance"])
        st.caption("Permutation importance: association with the model's predictions, NOT causal effect.")
    else:
        st.info("Run the pipeline first.")
with tabs[6]:
    if memory:
        rows = [dict(dataset=r["name"], best=max(r["scores"], key=r["scores"].get), best_score=max(r["scores"].values()),
                     **{k: round(v, 3) for k, v in list(r["meta"].items())[:6]}) for r in memory.records]
        st.dataframe(pd.DataFrame(rows))
    else:
        st.info("Memory is empty.")
with tabs[7]:
    st.write("Class-incremental stream on the digits data. Compares naive fine-tuning, replay and herding exemplar memory.")
    nt = st.slider("Tasks", 2, 5, 5); M = st.slider("Memory size", 20, 500, 200); lr = st.selectbox("Learner", ["softmax", "sgd", "mlp"]); lam = st.select_slider("EWC lambda (softmax only)", [10, 100, 300, 1000, 10000], 100)
    if st.button("Run continual-learning demo"):
        Xd, yd = load_dataset("digits")
        Xtr, Xte, ytr, yte = train_test_split(Xd.values, yd.values, test_size=0.3, stratify=yd, random_state=0)
        tasks = make_class_incremental(ytr, nt, 0)
        rows = {}
        for meth in ["naive", "reservoir", "herding", "joint"] + (["ewc"] if lr == "softmax" else []):
            r = run_stream(Xtr, ytr, Xte, yte, tasks, meth, lr, M, 8, 0, float(lam))
            rows[meth] = {k: round(v, 3) for k, v in r.items() if k != "R"}
        st.dataframe(pd.DataFrame(rows).T)
        st.caption("ACC = final average accuracy; BWT/Forgetting = backward transfer / average forgetting.")
