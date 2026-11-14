# AGENTIC SCOUT

Deterministic multi-agent AutoML for business data (**no LLM, no API keys**): EDA/profiling, task detection, preprocessing,
a kNN meta-learner over an experience memory, UCB1-bandit model search with a feedback loop, explanation, and continual learning (replay + EWC).
Code for the paper *Agentic Scout: An EDA-Driven Multi-Agent Framework for Automated Business Data Analysis with Meta-Learning, Bandit Search and Continual Learning* (Springer LNCS, Ubi-Media 2027).
Paper preview: `docs/paper_preview.pdf`. Architecture: `figures/agentic_scout_architecture.png`.

## Install
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt        # H2O needs Java 11+ (java -version)

## Try it (web app)
    python -m experiments.build_memory --datasets builtin    # optional: builds results/memory.json
    streamlit run app.py

## Use as a library
    from scout.orchestrator import ScoutAutoML
    model = ScoutAutoML(search="bandit", use_eda=True, use_meta=False, budget_evals=18, learn_online=False)
    model.fit(X, y)          # model.audit_trail lists every agent decision; model.predict(X)

## Reproduce the paper
Development benchmark (30 PMLB datasets, 3-fold, 1 seed):

    L="rf_default,hgb_default,logreg_default,scout_no_eda,scout_no_meta,scout_eda_meta,scout_bandit,scout_bandit_nometa"
    python -m experiments.run_benchmark --datasets pmlb:30 --methods $L --seeds 1 --folds 3 --n-jobs 2
    python -m experiments.run_benchmark --datasets pmlb:30 --methods h2o,flaml --budget 30 --seeds 1 --folds 3 --n-jobs 1
    python -m experiments.analyze

Held-out confirmation (16 unseen PMLB datasets, config fixed beforehand, rows capped at 2000):

    python -m experiments.run_benchmark --datasets "pmlb:<comma-separated names>" \
        --methods scout_bandit_nometa,scout_bandit,scout_ens,rf_default,hgb_default,logreg_default,h2o,flaml \
        --seeds 1 --folds 3 --max-rows 2000 --budget 30 --out results/heldout_ext.csv
    python -m experiments.heldout_summary         # -> results/table_heldout.csv, heldout_summary.json
    python -m experiments.fig_heldout             # -> results/fig_heldout.png (Fig. 3 of the paper)
    python -m experiments.fig_summary             # -> results/fig_summary.png (Fig. 4 of the paper)

The held-out dataset names are in `results/heldout_summary.json`. Other experiments: `experiments/run_continual.py`,
`run_meta_eval.py`, `online_memory.py`, `case_study.py`. Rebuild the paper (needs the Springer template `paper/lncs_template_clean.docx`,
not included): `python paper/make_paper.py`.

## Results in this repository
`results/*.csv|json` are the raw and aggregated results behind every number in the paper.
Honest summary: on the 30 development datasets Scout is on par with H2O AutoML and FLAML (no significant difference);
on the 16 held-out datasets it ranks first (2.78 vs 4.88 H2O, 5.31 FLAML; Holm-adjusted p = 0.047 and 0.009), suggestive but a small study.
The memory warm-start does not improve accuracy; it mainly reduces search cost.

## Layout
`scout/` agents, memory, continual learning, plots · `experiments/` runners and analysis · `paper/` LNCS paper generator · `app.py` Streamlit UI · `run_all.py` pipeline

## Citation
If you use this code, please cite the paper.
