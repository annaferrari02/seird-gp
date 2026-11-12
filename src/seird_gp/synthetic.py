"""Epidemie sintetiche con β(t) nota, per validare il GP (notebook 05).

Idea: si simula il SEIRD con una β(t) scelta da noi, si aggiunge rumore e si passa il risultato
alla stessa pipeline usata sui dati veri (γ(t) e μ(t) stimati dai dati sintetici, E0 stimato,
β_cost dalla baseline costante, stessa regola sul test). Così si vede che cosa il GP sa
ritrovare quando la risposta è nota.

Per essere vicini al caso reale:
- finestra, terminali (mobilità vera con i suoi ritardi correlati), stato iniziale (I0, R0, D0)
  e γ(t), μ(t) usati per generare i dati vengono dai dati veri della stessa finestra;
- il livello di β è quello della baseline costante sui dati veri (≈ 0.096 con σ = 1/4);
- rumore: binomiale negativa sui guariti e sui decessi giornalieri, log-normale sui positivi.

Scenari (fissati il 12/11/2026, PRIMA di guardare i risultati sintetici):
- A "mobilita": β = β_A · exp(1.5 · work_lag7). Ritardo vero 7, diverso dal 3 trovato sui dati
  veri, per non costruire uno scenario che dia ragione al risultato reale;
- B "tempo": β = β_B · exp(−0.2 · t), t in mesi: solo un trend nel tempo (controllo negativo
  per la mobilità, che nella seconda ondata è in parte correlata con t);
- C "restrizioni": β a gradini nelle date dei provvedimenti (data/restrizioni_trento.csv),
  nessun effetto diretto della mobilità (controllo negativo: restrizioni e mobilità cambiano
  insieme);
- D "costante": β costante (controllo: il GP deve restare su formule semplici).
In ogni scenario la costante (β_A, β_B, ...) è scelta in modo che la media geometrica di β
sul training sia uguale al livello dei dati veri: le epidemie hanno dimensioni realistiche.
"""

import json
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .data import RESULTS_DIR, TRENTO_POPULATION
from .fitting import data_scale, empirical_rates, normalized_fitness
from .gp import T_SCALE
from .model import COMPARTMENTS, SEIRDParams, simulate_fixed

SYNTH_DIR = RESULTS_DIR / "synthetic"
GP_SYNTH_DIR = RESULTS_DIR / "gp_synth"

# Rumore (12/11/2026): dispersione della binomiale negativa (varianza = m + m²/k) e
# deviazione standard del rumore log-normale sui positivi attuali.
NB_DISPERSION = 20.0
LOG_SD_POSITIVES = 0.05
E0_TRUE = 150.0  # ordine di grandezza di E0 stimato sulla seconda ondata (notebook 04)

# Gradini dello scenario C: moltiplicatori di β per livello di restrizione (fissati a priori:
# misure più dure = β più basso). Le date vengono dalla tabella dei provvedimenti.
RESTRICTION_FACTORS = {
    "misure nazionali": 0.85, "gialla": 0.75, "rossa/arancione": 0.55, "arancione": 0.65,
    "rossa": 0.5, "lockdown": 0.4, "bianca": 1.0,
}


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    # log β(t) a meno di una costante, giorno per giorno: (terminali X, date, tabella provvedimenti)
    shape: Callable[[pd.DataFrame, pd.DatetimeIndex, pd.DataFrame], np.ndarray]
    true_terminals: tuple  # terminali della formula vera (vuoto se nessuno)


def _restriction_steps(X, index, restrictions):
    """Gradini solo per i provvedimenti che iniziano dentro la finestra; prima del primo, fattore 1.
    (La tabella non registra le riaperture dopo il lockdown del 2020: a settembre 2020 non c'erano
    restrizioni comparabili, quindi il livello di partenza è 'nessuna restrizione'.)"""
    out = np.zeros(len(index))
    inside = restrictions[restrictions["inizio"] > index[0]].sort_values("inizio")
    for _, r in inside.iterrows():
        out[index >= r["inizio"]] = np.log(RESTRICTION_FACTORS[r["livello"]])
    return out


SCENARIOS = {
    "A": Scenario("mobilita", "β = c · exp(1.5 · work_lag7)",
                  lambda X, index, r: 1.5 * X["work_lag7"].to_numpy(float), ("work_lag7",)),
    "B": Scenario("tempo", "β = c · exp(−0.2 · t), t in mesi",
                  lambda X, index, r: -0.2 * np.arange(len(index)) / T_SCALE, ("t",)),
    "C": Scenario("restrizioni", "β = c · fattore del livello di restrizione (gradini)",
                  _restriction_steps, ()),
    "D": Scenario("costante", "β = c", lambda X, index, r: np.zeros(len(index)), ()),
}


def true_beta(scenario: Scenario, X: pd.DataFrame, index: pd.DatetimeIndex, n_train: int,
              beta_level: float, restrictions: pd.DataFrame | None = None) -> np.ndarray:
    """β vera giornaliera, con media geometrica sul training uguale a `beta_level`."""
    shape = np.asarray(scenario.shape(X, index, restrictions), float)
    return beta_level * np.exp(shape - shape[:n_train].mean())


def _nb(rng, mean, k=NB_DISPERSION):
    """Binomiale negativa con media `mean` e varianza mean + mean²/k (k grande: Poisson)."""
    mean = np.clip(mean, 0, None)
    p = k / (k + mean)
    return rng.negative_binomial(k, p)


def generate(real: pd.DataFrame, X: pd.DataFrame, start: str, end: str, beta: np.ndarray,
             sigma: float, E0: float = E0_TRUE, noise: bool = True, seed: int = 0,
             N: float = TRENTO_POPULATION) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dati sintetici della finestra con β vera `beta` (vettore giornaliero).

    Stato iniziale e γ(t), μ(t) per la simulazione vengono dai dati veri `real`.
    Restituisce (df, truth):
    - df: la serie `real` fino al giorno prima di `start`, poi i dati sintetici (stesse colonne
      di `real` che servono alla pipeline). I giorni prima servono a `empirical_rates`, come
      con i dati veri;
    - truth: S, E, I, R, D senza rumore, β, γ e μ veri, sui giorni della finestra.
    """
    obs = real.loc[start:end]
    rates = empirical_rates(real).loc[obs.index]
    first = obs.iloc[0]
    y0 = np.array([N - E0 - first["totale_positivi"] - first["dimessi_guariti"] - first["deceduti"],
                   E0, first["totale_positivi"], first["dimessi_guariti"], first["deceduti"]], float)
    y = simulate_fixed(beta, SEIRDParams(sigma, rates["gamma"].to_numpy(), rates["mu"].to_numpy()),
                       y0, len(obs))
    truth = pd.DataFrame(y, index=obs.index, columns=COMPARTMENTS)
    truth["beta"] = beta
    truth["gamma"], truth["mu"] = rates["gamma"].to_numpy(), rates["mu"].to_numpy()

    if noise:
        rng = np.random.default_rng(seed)
        new_R = _nb(rng, np.diff(truth["R"].to_numpy()))
        new_D = _nb(rng, np.diff(truth["D"].to_numpy()))
        R = first["dimessi_guariti"] + np.concatenate([[0], np.cumsum(new_R)])
        D = first["deceduti"] + np.concatenate([[0], np.cumsum(new_D)])
        I = truth["I"].to_numpy() * np.exp(rng.normal(0, LOG_SD_POSITIVES, len(obs)))
        I[0] = first["totale_positivi"]
        I = np.round(I)
    else:
        R, D, I = truth["R"].to_numpy(), truth["D"].to_numpy(), truth["I"].to_numpy()
    synth = pd.DataFrame({"totale_positivi": I, "dimessi_guariti": R, "deceduti": D}, index=obs.index)
    before = real.loc[:pd.Timestamp(start) - pd.Timedelta(days=1), list(synth.columns)]
    return pd.concat([before, synth]), truth


def oracle(problem, truth: pd.DataFrame) -> dict:
    """Errori della formula VERA, per sapere quanto errore resta anche con la β giusta.

    - "oracolo": dentro la pipeline (γ, μ stimati dai dati sintetici e congelati nel test,
      E0 stimato). È il meglio che il GP possa ottenere: il termine di confronto principale;
    - "oracolo, γ μ veri congelati": stessi tassi usati per generare i dati, congelati nel test
      come nella pipeline. La differenza con l'oracolo è l'errore dovuto alla STIMA di γ e μ;
    - "oracolo, γ μ veri": tassi veri anche nel test. La differenza con il precedente è
      l'errore dovuto al CONGELAMENTO dei tassi nel test;
    - "solo rumore": traiettoria vera senza rumore contro i dati rumorosi.
    """
    from .fitting import freeze_after
    from .gp import Problem

    g = np.log(truth["beta"].to_numpy() / problem.beta_ref)
    func = lambda t: g[np.rint(np.asarray(t) * T_SCALE).astype(int)]  # noqa: E731
    true_rates = truth[["gamma", "mu"]]
    variants = {
        "oracolo": problem,
        "oracolo, γ μ veri congelati": Problem(problem.obs, problem.n_train, problem.X, problem.sigma, None,
                                               freeze_after(true_rates, problem.obs.index[problem.n_train - 1]),
                                               problem.beta_ref),
        "oracolo, γ μ veri": Problem(problem.obs, problem.n_train, problem.X, problem.sigma, None,
                                     true_rates, problem.beta_ref),
    }
    out = {}
    for name, pr in variants.items():
        full = pr.evaluate_full(func, ["t"])
        out[f"{name} train"] = full["fitness_norm train"]
        out[f"{name} test"] = full["fitness_norm test"]
        out[f"{name} E0"] = full["E0"]
    obs, sim = problem.obs, truth[list(COMPARTMENTS)]
    scale = data_scale(obs.iloc[:problem.n_train])
    for period, sl in {"train": slice(None, problem.n_train), "test": slice(problem.n_train, None)}.items():
        out[f"solo rumore {period}"] = normalized_fitness(sim.iloc[sl], obs.iloc[sl], scale=scale)
    return out


def save_scenario(key: str, truth: pd.DataFrame, df: pd.DataFrame, meta: dict) -> None:
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    window = truth.index
    out = truth.add_suffix(" vero").join(df.loc[window[0]:window[-1]])
    out.to_csv(SYNTH_DIR / f"{key}.csv")
    (SYNTH_DIR / f"{key}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))


def load_meta(key: str) -> dict | None:
    path = SYNTH_DIR / f"{key}.json"
    return json.loads(path.read_text()) if path.exists() else None
