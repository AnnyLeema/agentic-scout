"""One command to reproduce every number in the paper, with independent stages running in parallel.

  python run_all.py --suite cc18:40 --cores 8 --budget 120 --retail "india retail sales dataset.csv" --target Returned

Stage graph:   build_memory ─┬─> benchmark (local methods) ─┐
                             ├─> meta_eval                  ├─> analyze -> make_paper
               continual ────┤                              │
               flaml ────────┘ (independent of memory)      │
               h2o  (separate JVM, run alone) ──────────────┘
Everything is resumable: re-running skips finished (dataset, seed, fold, method) rows.
"""
import argparse, subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="builtin", help="builtin | cc18 | cc18:N | labs | comma list")
ap.add_argument("--cores", type=int, default=4)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--budget", type=int, default=60, help="seconds per fit for FLAML / H2O")
ap.add_argument("--cl-datasets", default="digits", help="e.g. digits,openml:28,openml:554,openml:6")
ap.add_argument("--retail", default=None); ap.add_argument("--target", default="Returned")
ap.add_argument("--skip-h2o", action="store_true")
a = ap.parse_args()
py = sys.executable
def run(cmd):
    print(">>", cmd, flush=True); r = subprocess.run(cmd, shell=True); 
    if r.returncode: print("!! failed:", cmd)
os.makedirs("results", exist_ok=True)
half = max(a.cores // 2, 1)
run(f"{py} -m experiments.build_memory --datasets {a.suite} --n-jobs {a.cores}")
local = "rf_default,hgb_default,logreg_default,scout_no_eda,scout_no_meta,scout_eda_meta,scout_bandit,scout_bandit_nometa"
jobs = [f"{py} -m experiments.run_benchmark --datasets {a.suite} --methods {local} --seeds {a.seeds} --folds {a.folds} --n-jobs {half}",
        f"{py} -m experiments.run_meta_eval",
        f"{py} -m experiments.run_continual --datasets {a.cl_datasets} --seeds 5 --n-jobs {half}",
        f"{py} -m experiments.run_benchmark --datasets {a.suite} --methods flaml --budget {a.budget} --seeds {a.seeds} --folds {a.folds} --n-jobs {half}"]
with ThreadPoolExecutor(len(jobs)) as ex: list(ex.map(run, jobs))
if not a.skip_h2o:                                   # one JVM: run alone
    run(f"{py} -m experiments.run_benchmark --datasets {a.suite} --methods h2o --budget {a.budget} --seeds {a.seeds} --folds {a.folds} --n-jobs 1")
if a.retail: run(f'{py} -m experiments.case_study --csv "{a.retail}" --target {a.target}')
run(f"{py} -m experiments.analyze")
run(f"{py} paper/make_paper.py")
print("DONE -> paper/AgenticScout_v2_LNCS.docx  (resolve any red [TODO] markers, verify references)")
