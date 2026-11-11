"""Run del GP su una finestra: più configurazioni di terminali e più seed, in parallelo.

β_cost viene dalla baseline costante della stessa finestra e dello stesso σ
(results/04_baseline_all.pkl, prodotto da scripts/run_baselines.py); se manca, la si stima.

Uso (dalla cartella del repository, con l'ambiente attivo):
    python scripts/run_gp.py --start 2020-09-01 --end 2021-01-31 --sigma 5.2 \
        --configs tempo mobilita tempo+mobilita --seeds 1 2 3 --pop 200 --gen 50 --workers 2
I Pareto front si salvano in results/gp/<finestra>_s<σ>_<config>_seed<k>.json.
"""
import argparse
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from seird_gp.beta_models import constant_beta, fit_beta
from seird_gp.data import PROCESSED_DIR, RESULTS_DIR, load_mobility
from seird_gp.gp import CONFIGS, GP_RESULTS_DIR, GPConfig, knee, make_problem, run_gp, save_front
from seird_gp.inputs import build_gp_inputs

WINDOW_NAMES = {("2020-09-01", "2021-01-31"): "set20–gen21", ("2020-09-01", "2021-02-28"): "set20–feb21",
                ("2020-03-01", "2020-04-01"): "2020 (1/3–1/4)", ("2021-03-01", "2021-06-01"): "2021 (1/3–1/6)",
                ("2022-01-01", "2022-03-01"): "2022 (1/1–1/3)"}
SIGMA_LABELS = {5.2: "σ = 1/5.2", 3.0: "σ = 1/3 (latenza, Europa)", 4.0: "σ = 1/4 (latenza)",
                6.0: "σ = 1/6 (incubazione globale)", 6.4: "σ = 1/6.4 (meta-analisi)"}


def reference(df, start, end, sigma_days):
    """(E0, β_cost) della baseline costante (qui serve solo β_cost): dal file dei risultati se c'è, altrimenti stimati."""
    key = (WINDOW_NAMES.get((start, end)), "costante", SIGMA_LABELS.get(sigma_days))
    path = RESULTS_DIR / "04_baseline_all.pkl"
    if path.exists() and None not in key:
        res = pickle.load(open(path, "rb"))
        if key in res:
            p = res[key].params
            return p["E0"], p["beta"]
    r = fit_beta(df, start, end, constant_beta(), train_frac=2 / 3, fixed={"sigma": 1 / sigma_days},
                 rates_from_data=("gamma", "mu"))
    return r.params["E0"], r.params["beta"]


def job(args):
    problem, config, seed, cfg, out = args
    if out.exists():  # run già fatta: si può interrompere e riprendere
        print(f"{config} seed {seed}: già fatta, salto", flush=True)
        return out
    t0 = time.time()
    front, log = run_gp(problem, CONFIGS[config], cfg, seed=seed)
    save_front(front, out)
    k = knee(front)
    best = front["fitness_norm train"].idxmin()
    print(f"{config} seed {seed}: {time.time() - t0:.0f}s, {sum(r['nevals'] for r in log)} valutazioni, "
          f"front {len(front)}; gomito: {front.loc[k, 'formula']} "
          f"(train {front.loc[k, 'fitness_norm train']:.3f}, test {front.loc[k, 'fitness_norm test']:.3f}); "
          f"migliore: train {front.loc[best, 'fitness_norm train']:.3f}, test {front.loc[best, 'fitness_norm test']:.3f}",
          flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2020-09-01")
    p.add_argument("--end", default="2021-01-31")
    p.add_argument("--sigma", type=float, default=5.2, help="durata media in E (giorni): σ = 1/valore")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS))
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--pop", type=int, default=200)
    p.add_argument("--gen", type=int, default=50)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--tag", default="", help="prefisso dei file di output")
    p.add_argument("--max-abs-g", type=float, default=None,
                   help="vincolo di plausibilità: |g| ≤ valore su tutta la finestra (es. 1.5)")
    a = p.parse_args()

    df = pd.read_csv(PROCESSED_DIR / "trento_clean.csv", index_col=0, parse_dates=True)
    E0, beta_ref = reference(df, a.start, a.end, a.sigma)
    print(f"Finestra {a.start} – {a.end}, σ = 1/{a.sigma:g}: β_cost = {beta_ref:.4f} "
          f"(E0 stimato per ogni formula)", flush=True)
    problem = make_problem(df, build_gp_inputs(load_mobility()), a.start, a.end, 1 / a.sigma, beta_ref,
                           max_abs_g=a.max_abs_g)
    cfg = GPConfig(pop_size=a.pop, n_offspring=a.pop, n_gen=a.gen)
    name = f"{a.tag}{a.start}_{a.end}_s{a.sigma:g}"
    jobs = [(problem, c, s, cfg, GP_RESULTS_DIR / f"{name}_{c}_seed{s}.json") for c in a.configs for s in a.seeds]
    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(job, jobs))
    print("fatto")
