"""Validazione del GP su epidemie sintetiche con β(t) nota (notebook 05).

Per ogni scenario di `seird_gp.synthetic.SCENARIOS`:
1. genera i dati (mobilità, stato iniziale, γ(t), μ(t) e livello di β dai dati veri della finestra);
2. stima la baseline costante sui dati sintetici (β_cost, come per i dati veri);
3. lancia il GP con le stesse impostazioni del notebook 06, per ogni configurazione e seed.

Uso (dalla cartella del repository, con l'ambiente attivo):
    python scripts/run_synthetic.py --sigma 4 --scenarios A B C D --seeds 1 2 3 4 5 --workers 8
    (con --max-abs-g 2 --tag vinc_ per il vincolo di plausibilità, come nel notebook 06)
Risultati: results/synthetic/<scenario>.csv e .json (dati, verità, oracolo),
           results/gp_synth/<tag><scenario>_s<σ>_<config>_seed<k>.json (Pareto front).
Riprendibile: dati e run già salvati non si rifanno (salvo --regenerate per i dati).
"""
import argparse
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from seird_gp.beta_models import constant_beta, fit_beta, load_restrictions
from seird_gp.data import PROCESSED_DIR, RESULTS_DIR, load_mobility
from seird_gp.gp import CONFIGS, GPConfig, knee, make_problem, run_gp, save_front
from seird_gp.inputs import build_gp_inputs
from seird_gp.synthetic import (E0_TRUE, GP_SYNTH_DIR, SCENARIOS, SYNTH_DIR, generate, load_meta, oracle,
                                save_scenario, true_beta)

TRAIN_FRAC = 2 / 3
SIGMA_LABELS = {5.2: "σ = 1/5.2", 3.0: "σ = 1/3 (latenza, Europa)", 4.0: "σ = 1/4 (latenza)",
                6.0: "σ = 1/6 (incubazione globale)", 6.4: "σ = 1/6.4 (meta-analisi)"}
WINDOW_NAMES = {("2020-09-01", "2021-01-31"): "set20–gen21", ("2020-09-01", "2021-02-28"): "set20–feb21"}


def constant_level(df, start, end, sigma_days):
    """β della baseline costante stimata su `df` (stesse regole delle baseline del notebook 04)."""
    r = fit_beta(df, start, end, constant_beta(), train_frac=TRAIN_FRAC, fixed={"sigma": 1 / sigma_days},
                 rates_from_data=("gamma", "mu"))
    return float(r.params["beta"])


def real_level(real, start, end, sigma_days):
    """β della baseline costante sui dati veri: da results/04_baseline_all.pkl se c'è, altrimenti stimato."""
    key = (WINDOW_NAMES.get((start, end)), "costante", SIGMA_LABELS.get(sigma_days))
    path = RESULTS_DIR / "04_baseline_all.pkl"
    if path.exists() and None not in key:
        res = pickle.load(open(path, "rb"))
        if key in res:
            return float(res[key].params["beta"])
    return constant_level(real, start, end, sigma_days)


def prepare(args):
    """Genera i dati di uno scenario e stima β_cost. Restituisce il nome del file del problema."""
    key, real, X, start, end, sigma_days, level, noise_seed, regenerate = args
    meta = load_meta(key)
    same = meta is not None and meta["sigma_giorni"] == sigma_days and meta["finestra"] == [start, end] \
        and meta["seed rumore"] == noise_seed
    if same and not regenerate:
        print(f"Scenario {key}: dati già generati", flush=True)
        return key
    t0 = time.time()
    sc = SCENARIOS[key]
    index = real.loc[start:end].index
    n_train = max(2, int(round(len(index) * TRAIN_FRAC)))
    Xw = X.reindex(index)
    beta = true_beta(sc, Xw, index, n_train, level, load_restrictions())
    df, truth = generate(real, Xw, start, end, beta, 1 / sigma_days, seed=noise_seed)
    beta_ref = constant_level(df, start, end, sigma_days)
    problem = make_problem(df, X, start, end, 1 / sigma_days, beta_ref)
    meta = {"scenario": key, "nome": sc.name, "formula vera": sc.description,
            "terminali veri": list(sc.true_terminals), "finestra": [start, end], "sigma_giorni": sigma_days,
            "livello β (dati veri)": level, "β_cost (baseline costante sui dati sintetici)": beta_ref,
            "E0 vero": E0_TRUE, "seed rumore": noise_seed, **oracle(problem, truth)}
    save_scenario(key, truth, df, meta)
    pickle.dump(df, open(SYNTH_DIR / f"{key}_df.pkl", "wb"))
    print(f"Scenario {key} ({sc.name}): {time.time() - t0:.0f}s, β_cost = {beta_ref:.4f}, "
          f"oracolo train {meta['oracolo train']:.3f} test {meta['oracolo test']:.3f}", flush=True)
    return key


def job(args):
    key, config, seed, cfg, start, end, sigma_days, X, out, max_abs_g = args
    if out.exists():
        print(f"{key} {config} seed {seed}: già fatta, salto", flush=True)
        return out
    meta = load_meta(key)
    df = pickle.load(open(SYNTH_DIR / f"{key}_df.pkl", "rb"))
    problem = make_problem(df, X, start, end, 1 / sigma_days,
                           meta["β_cost (baseline costante sui dati sintetici)"], max_abs_g=max_abs_g)
    t0 = time.time()
    front, log = run_gp(problem, CONFIGS[config], cfg, seed=seed)
    save_front(front, out)
    k = knee(front)
    print(f"{key} {config} seed {seed}: {time.time() - t0:.0f}s, front {len(front)}; gomito: "
          f"{front.loc[k, 'formula']} (train {front.loc[k, 'fitness_norm train']:.3f}, "
          f"test {front.loc[k, 'fitness_norm test']:.3f}); oracolo test {meta['oracolo test']:.3f}", flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2020-09-01")
    p.add_argument("--end", default="2021-01-31")
    p.add_argument("--sigma", type=float, default=4.0, help="durata media in E (giorni): σ = 1/valore")
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    p.add_argument("--configs", nargs="+", default=list(CONFIGS))
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--pop", type=int, default=200)
    p.add_argument("--gen", type=int, default=100)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--regenerate", action="store_true", help="rigenera i dati sintetici")
    p.add_argument("--tag", default="", help="prefisso dei file dei Pareto front")
    p.add_argument("--max-abs-g", type=float, default=None,
                   help="vincolo di plausibilità: |g| ≤ valore su tutta la finestra (es. 2)")
    a = p.parse_args()

    real = pd.read_csv(PROCESSED_DIR / "trento_clean.csv", index_col=0, parse_dates=True)
    X = build_gp_inputs(load_mobility())
    level = real_level(real, a.start, a.end, a.sigma)
    print(f"Livello di β dai dati veri (baseline costante, σ = 1/{a.sigma:g}): {level:.4f}", flush=True)

    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(prepare, [(k, real, X, a.start, a.end, a.sigma, level, a.noise_seed, a.regenerate)
                              for k in a.scenarios]))
        cfg = GPConfig(pop_size=a.pop, n_offspring=a.pop, n_gen=a.gen)
        jobs = [(k, c, s, cfg, a.start, a.end, a.sigma, X,
                 GP_SYNTH_DIR / f"{a.tag}{k}_s{a.sigma:g}_{c}_seed{s}.json", a.max_abs_g)
                for k in a.scenarios for c in a.configs for s in a.seeds]
        list(ex.map(job, jobs))
    print("fatto")
