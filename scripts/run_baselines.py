"""Tutte le baseline su tutte le finestre, impostazione di riferimento di Anna (sezione 6 del 02)."""
import pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from seird_gp.data import PROCESSED_DIR, RESULTS_DIR, load_mobility
from seird_gp.fitting import SIGMA_LIT
from seird_gp.inputs import build_gp_inputs
from seird_gp.beta_models import (constant_beta, fit_beta, linear_mobility_beta,
                                  load_restrictions, piecewise_beta)

WINDOWS = {"2020 (1/3–1/4)": ("2020-03-01", "2020-04-01"),
           "2021 (1/3–1/6)": ("2021-03-01", "2021-06-01"),
           "2022 (1/1–1/3)": ("2022-01-01", "2022-03-01"),
           "set20–gen21": ("2020-09-01", "2021-01-31"),
           "set20–feb21": ("2020-09-01", "2021-02-28")}
data = pd.read_csv(PROCESSED_DIR / "trento_clean.csv", index_col=0, parse_dates=True)
inputs = build_gp_inputs(load_mobility())
breaks = load_restrictions()["inizio"]
MODELS = {"costante": constant_beta(), "a tratti": piecewise_beta(breaks),
          "lineare retail": linear_mobility_beta(inputs, ["retail_lag0"]),
          "lineare work": linear_mobility_beta(inputs, ["work_lag0"]),
          "lineare retail+work": linear_mobility_beta(inputs, ["retail_lag0", "work_lag0"])}
SIGMAS = {"σ = 1/5.2": SIGMA_LIT, "σ = 1/4 (latenza)": 1 / 4, "σ = 1/6.4 (meta-analisi)": 1 / 6.4}

def job(args):
    w, m, s = args
    a, b = WINDOWS[w]; t0 = time.time()
    r = fit_beta(data, a, b, MODELS[m], train_frac=2 / 3, fixed={"sigma": SIGMAS[s]},
                 rates_from_data=("gamma", "mu"))
    r.extra["rates"] = None
    print(f"{w} | {m} | {s}: {time.time()-t0:.0f}s, test {r.rmsd.loc['test','fitness_norm']:.3f}", flush=True)
    return (w, m, s), r

if __name__ == "__main__":
    jobs = [(w, m, "σ = 1/5.2") for w in WINDOWS for m in MODELS]
    jobs += [(w, "costante", s) for w in WINDOWS for s in list(SIGMAS)[1:]]
    with ProcessPoolExecutor(int(sys.argv[1])) as ex:
        res = dict(ex.map(job, jobs))
    RESULTS_DIR.mkdir(exist_ok=True)
    pickle.dump(res, open(RESULTS_DIR / "04_baseline_all.pkl", "wb"))
    print("fatto")
