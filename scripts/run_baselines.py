"""Tutte le baseline su tutte le finestre, impostazione di riferimento di Anna (notebook 02, sezione 6):
γ(t) e μ(t) dai dati, fitness normalizzata, training sui primi 2/3 della finestra, dati `clean`.

- 5 forme di β (costante, a tratti, 3 lineari nella mobilità) con σ = 1/5.2;
- sensibilità su σ per β costante (σ = 1/3, 1/4, 1/5.2, 1/6, 1/6.4).

Uso (dalla cartella del repository, con l'ambiente attivo):
    python scripts/run_baselines.py [n_processi] [--redo "finestra|baseline" ...]
I fit già presenti in results/04_baseline_all.pkl non si rifanno, salvo quelli in --redo.
Circa 2 ore con 2 core la prima volta.
"""
import argparse
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from seird_gp.beta_models import (constant_beta, fit_beta, linear_mobility_beta,
                                  load_restrictions, piecewise_beta)
from seird_gp.data import PROCESSED_DIR, RESULTS_DIR, load_mobility
from seird_gp.fitting import SIGMA_LIT
from seird_gp.inputs import build_gp_inputs

OUT = RESULTS_DIR / "04_baseline_all.pkl"
WINDOWS = {"2020 (1/3–1/4)": ("2020-03-01", "2020-04-01"),
           "2021 (1/3–1/6)": ("2021-03-01", "2021-06-01"),
           "2022 (1/1–1/3)": ("2022-01-01", "2022-03-01"),
           "set20–gen21": ("2020-09-01", "2021-01-31"),
           "set20–feb21": ("2020-09-01", "2021-02-28")}
REF = "σ = 1/5.2"
SIGMAS = {REF: SIGMA_LIT,
          "σ = 1/3 (latenza, Europa)": 1 / 3,
          "σ = 1/4 (latenza)": 1 / 4,
          "σ = 1/6 (incubazione globale)": 1 / 6,
          "σ = 1/6.4 (meta-analisi)": 1 / 6.4}

data = pd.read_csv(PROCESSED_DIR / "trento_clean.csv", index_col=0, parse_dates=True)
inputs = build_gp_inputs(load_mobility())
MODELS = {"costante": constant_beta(),
          "a tratti": piecewise_beta(load_restrictions()["inizio"]),
          "lineare retail": linear_mobility_beta(inputs, ["retail_lag0"]),
          "lineare work": linear_mobility_beta(inputs, ["work_lag0"]),
          "lineare retail+work": linear_mobility_beta(inputs, ["retail_lag0", "work_lag0"])}


def job(key):
    w, m, s = key
    a, b = WINDOWS[w]
    t0 = time.time()
    r = fit_beta(data, a, b, MODELS[m], train_frac=2 / 3, fixed={"sigma": SIGMAS[s]},
                 rates_from_data=("gamma", "mu"))
    r.extra["rates"] = None  # si ricalcolano con empirical_rates; non serve salvarli
    print(f"{w} | {m} | {s}: {time.time() - t0:.0f}s, test {r.rmsd.loc['test', 'fitness_norm']:.3f}",
          flush=True)
    return key, r


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("workers", nargs="?", type=int, default=2)
    p.add_argument("--redo", nargs="*", default=[], help='coppie "finestra|baseline" da rifare')
    p.add_argument("--all-models-sigma", nargs="*", default=[], metavar="FINESTRA|σ",
                   help='tutte le baseline anche per altri σ, es. "set20–gen21|σ = 1/4 (latenza)"')
    args = p.parse_args()

    keys = [(w, m, REF) for w in WINDOWS for m in MODELS]
    keys += [(w, "costante", s) for w in WINDOWS for s in SIGMAS if s != REF]
    for item in args.all_models_sigma:  # serve per confrontare il GP alla pari con altri σ
        w, s = item.split("|")
        keys += [(w, m, s) for m in MODELS if (w, m, s) not in keys]
    done = pickle.load(open(OUT, "rb")) if OUT.exists() else {}
    redo = {tuple(x.split("|")) for x in args.redo}
    todo = [k for k in keys if k not in done or (k[0], k[1]) in redo]
    print(f"{len(todo)} fit da fare, {len(keys) - len(todo)} già fatti", flush=True)

    with ProcessPoolExecutor(args.workers) as ex:
        for key, r in ex.map(job, todo):
            done[key] = r
            RESULTS_DIR.mkdir(exist_ok=True)
            pickle.dump(done, open(OUT, "wb"))  # salva dopo ogni fit: si può interrompere
    print("fatto")
