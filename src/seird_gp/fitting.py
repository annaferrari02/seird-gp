"""Fitting del SEIRD con β costante (baseline) su una finestra temporale.

Fitness di default: 0.9 * RMSD(decessi) + 0.1 * RMSD(infetti), dove D si confronta con
`deceduti` e I con `totale_positivi`. È quella descritta nel testo di Pianesi e Pojer;
il loro codice però calcola 0.9 * RMSD(decessi) + 0.1 * RMSD(guariti), vedi
`pianesi_pojer_fitness`.
In alternativa `normalized_fitness`: ogni RMSD diviso per la deviazione standard dei dati
di training, così i due termini hanno la stessa scala.
Ottimizzatore: differential evolution (scipy), un algoritmo evolutivo.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from .data import TRENTO_POPULATION
from .model import COMPARTMENTS, SEIRDParams, simulate

WEIGHTS = {"deceduti": 0.9, "totale_positivi": 0.1}  # come nel testo del paper, non nel codice
PARAM_NAMES = ["beta", "sigma", "gamma", "mu", "E0"]

# Intervalli plausibili (tassi giornalieri)
DEFAULT_BOUNDS = {
    "beta": (0.01, 2.0),
    "sigma": (1 / 14, 1 / 2),   # incubazione 2-14 giorni
    "gamma": (1 / 60, 1 / 3),   # uscita da I verso R: 3-60 giorni (i guariti DPC sono lenti)
    "mu": (0.0, 0.05),
}

# Parametri da letteratura
# Incubazione media 5.2 giorni (IC 95% 4.1-7.0): Li et al., NEJM 2020, doi:10.1056/NEJMoa2001316
SIGMA_LIT = 1 / 5.2

# Pesi della fitness normalizzata: nessuna preferenza a priori fra decessi e positivi
NORM_WEIGHTS = {"deceduti": 0.5, "totale_positivi": 0.5}


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


RATE_COLUMNS = {"gamma": "dimessi_guariti", "mu": "deceduti"}


def empirical_rates(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """γ(t) e μ(t) letti dai dati: dR/dt = γ I e dD/dt = μ I.

    γ(t) = guariti nuovi negli ultimi `window` giorni / somma dei positivi negli stessi giorni
    (idem μ con i decessi). Media mobile solo all'indietro: il valore di un giorno usa
    soltanto dati fino a quel giorno. Giorni senza positivi: ultimo valore valido.
    `df` deve essere la serie intera, così l'inizio di una finestra usa anche i giorni prima.
    """
    # i nuovi del giorno k nascono nell'intervallo [k-1, k]: positivi medi su quell'intervallo
    mid = (df["totale_positivi"] + df["totale_positivi"].shift(1).fillna(df["totale_positivi"])) / 2
    positives = mid.rolling(window, min_periods=1).sum()
    out = {}
    for name, col in RATE_COLUMNS.items():
        new = df[col].diff().clip(lower=0).rolling(window, min_periods=1).sum()
        out[name] = (new / positives.where(positives > 0)).ffill().bfill().fillna(0.0)
    return pd.DataFrame(out, index=df.index)


def freeze_after(rates: pd.DataFrame, last_known: pd.Timestamp) -> pd.DataFrame:
    """Dopo `last_known` i tassi restano all'ultimo valore noto (niente dati del futuro nel test)."""
    out = rates.copy()
    out.loc[out.index > last_known] = out.loc[:last_known].iloc[-1].values
    return out


def _daily(values: np.ndarray):
    """Funzione del tempo (giorni dall'inizio finestra), lineare fra i valori giornalieri.

    Lineare e non a gradini: senza salti l'integratore fa meno passi.
    """
    x = np.arange(len(values), dtype=float)
    return lambda t: float(np.interp(t, x, values))


def simulate_window(theta, obs: pd.DataFrame, N: float = TRENTO_POPULATION,
                    rates: pd.DataFrame | None = None) -> pd.DataFrame:
    """Simula la finestra. Con `rates` (colonne gamma e/o mu, stesso indice di `obs`)
    quei tassi variano nel tempo e i valori corrispondenti in `theta` sono ignorati."""
    beta, sigma, gamma, mu, E0 = theta
    if rates is not None:
        if "gamma" in rates:
            gamma = _daily(rates["gamma"].to_numpy())
        if "mu" in rates:
            mu = _daily(rates["mu"].to_numpy())
    t = np.arange(len(obs), dtype=float)
    y = simulate(lambda _: beta, SEIRDParams(sigma, gamma, mu), initial_state(obs, E0, N), t)
    return pd.DataFrame(y, index=obs.index, columns=COMPARTMENTS)


def rmsd(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def fitness(sim: pd.DataFrame, obs: pd.DataFrame, weights=WEIGHTS) -> float:
    return (weights["deceduti"] * rmsd(sim["D"], obs["deceduti"])
            + weights["totale_positivi"] * rmsd(sim["I"], obs["totale_positivi"]))


def pianesi_pojer_fitness(sim: pd.DataFrame, obs: pd.DataFrame) -> float:
    """Fitness calcolata davvero nel codice di Pianesi e Pojer: decessi e guariti.

    Repo LunaBaozi/SEIRD-covid-models-with-GAs, SEIRD_model/seird_problem.py e runner.py:
    obiettivi [RMSE_D, RMSE_R, RMSE_I], pesi [0.9, 0.1] applicati ai primi due.
    Con i loro parametri riproduce il 12.50 della loro Tabella I. Serve solo per il
    confronto con il paper.
    """
    return 0.9 * rmsd(sim["D"], obs["deceduti"]) + 0.1 * rmsd(sim["R"], obs["dimessi_guariti"])


def data_scale(obs: pd.DataFrame, cols=tuple(NORM_WEIGHTS)) -> dict:
    """Deviazione standard di ogni serie osservata (minimo 1, per serie quasi piatte)."""
    return {c: max(float(obs[c].std()), 1.0) for c in cols}


def normalized_fitness(sim: pd.DataFrame, obs: pd.DataFrame, weights=NORM_WEIGHTS,
                       scale: dict | None = None) -> float:
    """Media pesata di RMSD(D)/std(decessi) e RMSD(I)/std(positivi).

    Durante il fit `obs` è il training, quindi la scala è quella del training.
    Per valutare il test passare `scale` calcolata sul training, così train e test
    sono misurati con lo stesso metro.
    """
    scale = scale or data_scale(obs, tuple(weights))
    return (weights["deceduti"] * rmsd(sim["D"], obs["deceduti"]) / scale["deceduti"]
            + weights["totale_positivi"] * rmsd(sim["I"], obs["totale_positivi"])
            / scale["totale_positivi"])


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
                 fitness_fn=None, maxiter: int = 300, rates_from_data: tuple = (),
                 N: float = TRENTO_POPULATION) -> FitResult:
    """Stima β, σ, γ, μ, E0 sulla prima `train_frac` della finestra [start, end].

    La simulazione copre tutta la finestra: i giorni dopo il training servono come test.
    `fixed` blocca dei parametri a un valore (es. {"sigma": 1/5.2});
    `fitness_fn` sostituisce la fitness usata per l'ottimizzazione.
    `rates_from_data` ("gamma", "mu" o entrambi): quei tassi non si stimano ma si leggono
    dai dati con `empirical_rates`, congelati all'ultimo giorno di training nel test.
    In `params` compare la loro media sul training, usata da `derived()`.
    """
    fitness_fn = fitness_fn or fitness
    obs = df.loc[start:end]
    n_train = max(2, int(round(len(obs) * train_frac)))
    train = obs.iloc[:n_train]

    rates = None
    if rates_from_data:
        rates = empirical_rates(df).loc[obs.index, list(rates_from_data)]
        rates = freeze_after(rates, train.index[-1])
        fixed = {**(fixed or {}), **rates.iloc[:n_train].mean().to_dict()}

    b = {**DEFAULT_BOUNDS, **(bounds or {})}
    I0 = obs["totale_positivi"].iloc[0]
    b.setdefault("E0", (0.0, 10 * max(I0, 50)))
    for k, v in (fixed or {}).items():
        b[k] = (v, v)
    box = [b[k] for k in PARAM_NAMES]
    train_rates = None if rates is None else rates.iloc[:n_train]

    def objective(theta):
        try:
            return fitness_fn(simulate_window(theta, train, N, train_rates), train)
        except RuntimeError:
            return 1e12

    res = differential_evolution(objective, box, seed=seed, maxiter=maxiter,
                                 tol=1e-7, polish=True)
    sim = simulate_window(res.x, obs, N, rates)

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
    return FitResult(
        params=dict(zip(PARAM_NAMES, res.x)),
        train_end=train.index[-1],
        fitness_train=float(res.fun),
        rmsd=pd.DataFrame(table).T,
        simulation=sim,
        extra={"n_eval": res.nfev, "success": res.success,
               "at_bound": at_bounds(dict(zip(PARAM_NAMES, res.x)), b, fixed),
               "rates": rates},
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