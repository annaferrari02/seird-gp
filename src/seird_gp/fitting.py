"""Fitting del SEIRD con β costante (baseline) su una finestra temporale.

Fitness come in Pianesi e Pojer: 0.9 * RMSD(decessi) + 0.1 * RMSD(infetti),
dove D si confronta con `deceduti` e I con `totale_positivi`.
Ottimizzatore: differential evolution (scipy), un algoritmo evolutivo.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from .data import TRENTO_POPULATION
from .model import COMPARTMENTS, SEIRDParams, simulate

WEIGHTS = {"deceduti": 0.9, "totale_positivi": 0.1}
PARAM_NAMES = ["beta", "sigma", "gamma", "mu", "E0"]

# Intervalli plausibili (tassi giornalieri)
DEFAULT_BOUNDS = {
    "beta": (0.01, 2.0),
    "sigma": (1 / 14, 1 / 2),   # incubazione 2-14 giorni
    "gamma": (1 / 60, 1 / 3),   # uscita da I verso R: 3-60 giorni (i guariti DPC sono lenti)
    "mu": (0.0, 0.05),
}


@dataclass
class FitResult:
    params: dict
    train_end: pd.Timestamp
    fitness_train: float
    rmsd: pd.DataFrame          # RMSD per compartimento e per periodo (train/test)
    simulation: pd.DataFrame    # S, E, I, R, D simulati, indicizzati per data
    extra: dict = field(default_factory=dict)

    def derived(self) -> dict:
        """Grandezze confrontabili con la Tabella I di Pianesi e Pojer."""
        b, g, m = self.params["beta"], self.params["gamma"], self.params["mu"]
        return {"R0": b / (g + m), "t_inf": 1 / (g + m), "f": m / (g + m)}


def initial_state(obs: pd.DataFrame, E0: float, N: float = TRENTO_POPULATION) -> np.ndarray:
    """Stato iniziale dal primo giorno osservato; E0 è stimato perché non osservato."""
    first = obs.iloc[0]
    I0, R0, D0 = first["totale_positivi"], first["dimessi_guariti"], first["deceduti"]
    return np.array([N - E0 - I0 - R0 - D0, E0, I0, R0, D0], dtype=float)


def simulate_window(theta, obs: pd.DataFrame, N: float = TRENTO_POPULATION) -> pd.DataFrame:
    beta, sigma, gamma, mu, E0 = theta
    t = np.arange(len(obs), dtype=float)
    y = simulate(lambda _: beta, SEIRDParams(sigma, gamma, mu), initial_state(obs, E0, N), t)
    return pd.DataFrame(y, index=obs.index, columns=COMPARTMENTS)


def rmsd(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def fitness(sim: pd.DataFrame, obs: pd.DataFrame, weights=WEIGHTS) -> float:
    return (weights["deceduti"] * rmsd(sim["D"], obs["deceduti"])
            + weights["totale_positivi"] * rmsd(sim["I"], obs["totale_positivi"]))


def relative_fitness(sim: pd.DataFrame, obs: pd.DataFrame) -> float:
    """Variante: errori divisi per la deviazione standard dei dati e guariti inclusi.

    Evita che il termine sugli infetti (migliaia) domini quello sui decessi (decine).
    """
    def term(model, data):
        return rmsd(sim[model], obs[data]) / max(obs[data].std(), 1.0)
    return (0.6 * term("D", "deceduti") + 0.2 * term("I", "totale_positivi")
            + 0.2 * term("R", "dimessi_guariti"))


def fit_baseline(df: pd.DataFrame, start: str, end: str, train_frac: float = 1.0,
                 seed: int = 0, bounds: dict | None = None, fixed: dict | None = None,
                 fitness_fn=None, maxiter: int = 300,
                 N: float = TRENTO_POPULATION) -> FitResult:
    """Stima β, σ, γ, μ, E0 sulla prima `train_frac` della finestra [start, end].

    La simulazione copre tutta la finestra: i giorni dopo il training servono come test.
    `fixed` blocca dei parametri a un valore (es. {"sigma": 1/5.2});
    `fitness_fn` sostituisce la fitness usata per l'ottimizzazione.
    """
    fitness_fn = fitness_fn or fitness
    obs = df.loc[start:end]
    n_train = max(2, int(round(len(obs) * train_frac)))
    train = obs.iloc[:n_train]

    b = {**DEFAULT_BOUNDS, **(bounds or {})}
    I0 = obs["totale_positivi"].iloc[0]
    b.setdefault("E0", (0.0, 10 * max(I0, 50)))
    for k, v in (fixed or {}).items():
        b[k] = (v, v)
    box = [b[k] for k in PARAM_NAMES]

    def objective(theta):
        try:
            return fitness_fn(simulate_window(theta, train, N), train)
        except RuntimeError:
            return 1e12

    res = differential_evolution(objective, box, seed=seed, maxiter=maxiter,
                                 tol=1e-7, polish=True)
    sim = simulate_window(res.x, obs, N)

    periods = {"train": slice(None, n_train), "test": slice(n_train, None)}
    table = {}
    for period, sl in periods.items():
        if len(obs.iloc[sl]) == 0:
            continue
        table[period] = {
            "D": rmsd(sim["D"].iloc[sl], obs["deceduti"].iloc[sl]),
            "I": rmsd(sim["I"].iloc[sl], obs["totale_positivi"].iloc[sl]),
            "fitness": fitness(sim.iloc[sl], obs.iloc[sl]),
        }
    return FitResult(
        params=dict(zip(PARAM_NAMES, res.x)),
        train_end=train.index[-1],
        fitness_train=float(res.fun),
        rmsd=pd.DataFrame(table).T,
        simulation=sim,
        extra={"n_eval": res.nfev, "success": res.success,
               "at_bound": at_bounds(dict(zip(PARAM_NAMES, res.x)), b, fixed)},
    )


def at_bounds(params: dict, bounds: dict, fixed: dict | None = None, tol: float = 0.01) -> list:
    """Parametri stimati che sono finiti sul bordo dell'intervallo (segnale di non identificabilità)."""
    out = []
    for k, v in params.items():
        if fixed and k in fixed:
            continue
        lo, hi = bounds[k]
        if v <= lo + tol * (hi - lo) or v >= hi - tol * (hi - lo):
            out.append(k)
    return out
