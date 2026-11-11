"""Baseline con β(t) variabile ma di forma fissata, da confrontare con il GP.

Tre forme, come nella fase 2 del documento di progetto:
- `constant_beta`: β costante (stessa ipotesi di `fit_baseline`, utile come controllo);
- `piecewise_beta`: un β per ogni periodo fra due provvedimenti (date in `data/restrizioni_trento.csv`);
- `linear_mobility_beta`: β(t) = β0 · (1 + Σ c_k · x_k(t)), con x_k gli input di mobilità del GP.

`fit_beta` stima σ, γ, μ, E0 (o li fissa / li legge dai dati, come `fit_baseline`) insieme ai
parametri della forma di β. Regola comune con il GP: nel test non si usa nessun dato futuro
degli *esiti* (positivi, decessi, guariti). Gli input esterni noti in anticipo o osservati
indipendentemente dall'epidemia (date dei provvedimenti, mobilità) si usano anche nel test,
esattamente come farà il GP con m(t).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from .data import ROOT, TRENTO_POPULATION
from .fitting import (
    DEFAULT_BOUNDS, FitResult, at_bounds, data_scale, empirical_rates, fitness, freeze_after,
    normalized_fitness, rmsd, simulate_window,
)

RESTRICTIONS_PATH = ROOT / "data" / "restrizioni_trento.csv"
BETA_BOUNDS = DEFAULT_BOUNDS["beta"]
COEF_BOUNDS = (-3.0, 5.0)  # c_k: segno libero, i dati decidono se più mobilità = più contagio


@dataclass(frozen=True)
class BetaModel:
    """Forma di β(t). `build(valori, indice_finestra, fine_training)` restituisce β come
    funzione dei giorni dall'inizio della finestra."""
    name: str
    params: tuple
    bounds: dict
    build: Callable[[dict, pd.DatetimeIndex, pd.Timestamp], Callable[[float], float]]
    # parametri stimabili con il training (default: tutti)
    active: Callable[[pd.DatetimeIndex, pd.Timestamp], tuple] | None = None

    def active_params(self, index: pd.DatetimeIndex, train_end: pd.Timestamp) -> tuple:
        return self.params if self.active is None else self.active(index, train_end)


def constant_beta() -> BetaModel:
    return BetaModel("costante", ("beta",), {"beta": BETA_BOUNDS},
                     lambda v, index, train_end: (lambda t, b=v["beta"]: b))


def load_restrictions(path: Path = RESTRICTIONS_PATH) -> pd.DataFrame:
    """Tabella dei provvedimenti: una riga per ogni inizio di periodo."""
    df = pd.read_csv(path, parse_dates=["inizio"], comment="#")
    return df.sort_values("inizio").reset_index(drop=True)


MIN_TRAIN_DAYS = 7


def piecewise_beta(breaks, min_train_days: int = MIN_TRAIN_DAYS) -> BetaModel:
    """Un β per ogni periodo. `breaks` = date di inizio dei periodi dopo il primo.

    Un periodo ha un suo β solo se ha almeno `min_train_days` giorni nel training.
    Altrimenti resta al β del periodo precedente: i periodi che iniziano nel test non hanno
    dati, e quelli che iniziano a ridosso della fine del training ne hanno troppo pochi
    (con 5 giorni, finestra 2021, il β stimato era instabile e si trascinava in tutto il test).
    I parametri si chiamano beta_0, beta_1, ...; quelli dei periodi non stimabili non si usano.
    """
    breaks = pd.DatetimeIndex(sorted(pd.to_datetime(list(breaks))))
    names = tuple(f"beta_{k}" for k in range(len(breaks) + 1))

    def inside(index, train_end):
        """Inizi di periodo dentro il training con almeno `min_train_days` giorni di dati."""
        return [b for b in breaks
                if index[0] < b and (train_end - b).days + 1 >= min_train_days]

    def build(v, index, train_end):
        offsets = np.array([(b - index[0]).days for b in inside(index, train_end)], float)
        values = np.array([v[f"beta_{k}"] for k in range(len(offsets) + 1)])

        def beta(t):
            return float(values[np.searchsorted(offsets, t, side="right")])
        return beta

    def active(index, train_end):
        return names[:len(inside(index, train_end)) + 1]

    return BetaModel("a tratti", names, {n: BETA_BOUNDS for n in names}, build, active)


def linear_mobility_beta(inputs: pd.DataFrame, columns) -> BetaModel:
    """β(t) = max(0, β0 · (1 + Σ c_k x_k(t))), x_k = colonne di `build_gp_inputs`.

    Con gli input normalizzati di Anna (−0.4 = −40% rispetto a gen–feb 2020), β0 è il β a
    mobilità di baseline. Il taglio a 0 è lo stesso vincolo β ≥ 0 previsto per il GP.
    """
    columns = list(columns)
    names = ("beta0",) + tuple(f"c_{c}" for c in columns)
    bounds = {"beta0": BETA_BOUNDS, **{f"c_{c}": COEF_BOUNDS for c in columns}}

    def build(v, index, train_end):
        x = inputs.reindex(index)[columns]
        if x.isna().any().any():
            raise ValueError("Input di mobilità mancanti nella finestra")
        values = v["beta0"] * (1 + x.to_numpy() @ np.array([v[f"c_{c}"] for c in columns]))
        values = np.clip(values, 0.0, None)
        days = np.arange(len(index), dtype=float)
        return lambda t: float(np.interp(t, days, values))

    return BetaModel(f"lineare in {', '.join(columns)}", names, bounds, build)


def fit_beta(df: pd.DataFrame, start: str, end: str, model: BetaModel, train_frac: float = 1.0,
             seed: int = 0, bounds: dict | None = None, fixed: dict | None = None,
             fitness_fn=None, maxiter: int = 300, rates_from_data: tuple = (),
             N: float = TRENTO_POPULATION) -> FitResult:
    """Come `fit_baseline`, ma con β(t) della forma `model`.

    Parametri stimati: σ, γ, μ, E0 (salvo `fixed` / `rates_from_data`) e quelli di `model`.
    In `params["beta"]` c'è la media di β(t) sui giorni di training, così `derived()` funziona.
    In `extra["beta"]` c'è la serie β(t) su tutta la finestra.
    """
    fitness_fn = fitness_fn or normalized_fitness
    obs = df.loc[start:end]
    n_train = max(2, int(round(len(obs) * train_frac)))
    train = obs.iloc[:n_train]
    train_end = train.index[-1]

    used = list(model.active_params(obs.index, train_end))

    rates = None
    fixed = dict(fixed or {})
    if rates_from_data:
        rates = empirical_rates(df).loc[obs.index, list(rates_from_data)]
        rates = freeze_after(rates, train_end)
        fixed.update(rates.iloc[:n_train].mean().to_dict())

    b = {**DEFAULT_BOUNDS, **model.bounds, **(bounds or {})}
    b.setdefault("E0", (0.0, 10 * max(obs["totale_positivi"].iloc[0], 50)))
    for k, v in fixed.items():
        b[k] = (v, v)
    names = ["sigma", "gamma", "mu", "E0"] + used
    box = [b[k] for k in names]
    train_rates = None if rates is None else rates.iloc[:n_train]

    def beta_fn(theta):
        v = dict(zip(names, theta))
        return model.build({**{p: np.nan for p in model.params}, **v}, obs.index, train_end)

    def run(theta, data, r):
        v = dict(zip(names, theta))
        return simulate_window((np.nan, v["sigma"], v["gamma"], v["mu"], v["E0"]), data, N, r,
                               beta_fn=beta_fn(theta))

    def objective(theta):
        try:
            return fitness_fn(run(theta, train, train_rates), train)
        except RuntimeError:
            return 1e12

    res = differential_evolution(objective, box, seed=seed, maxiter=maxiter, tol=1e-7, polish=True)
    sim = run(res.x, obs, rates)
    f = beta_fn(res.x)
    beta_series = pd.Series([f(t) for t in range(len(obs))], index=obs.index, name="beta")

    periods = {"train": slice(None, n_train), "test": slice(n_train, None)}
    train_scale = data_scale(train)
    table = {}
    for period, sl in periods.items():
        if len(obs.iloc[sl]) == 0:
            continue
        table[period] = {
            "D": rmsd(sim["D"].iloc[sl], obs["deceduti"].iloc[sl]),
            "I": rmsd(sim["I"].iloc[sl], obs["totale_positivi"].iloc[sl]),
            "fitness": fitness(sim.iloc[sl], obs.iloc[sl]),
            "fitness_norm": normalized_fitness(sim.iloc[sl], obs.iloc[sl], scale=train_scale),
        }
    params = dict(zip(names, res.x))
    params["beta"] = float(beta_series.iloc[:n_train].mean())
    return FitResult(
        params=params,
        train_end=train_end,
        fitness_train=float(res.fun),
        rmsd=pd.DataFrame(table).T,
        simulation=sim,
        extra={"n_eval": res.nfev, "success": res.success, "model": model.name,
               "n_params": len([n for n in names if b[n][0] != b[n][1]]),
               "at_bound": at_bounds(dict(zip(names, res.x)), b, fixed),
               "rates": rates, "beta": beta_series},
    )
