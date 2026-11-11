"""Modello SEIRD con tasso di trasmissione β(t) variabile nel tempo."""

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np
from scipy.integrate import solve_ivp

COMPARTMENTS = ("S", "E", "I", "R", "D")


@dataclass(frozen=True)
class SEIRDParams:
    """Parametri costanti del modello (tassi giornalieri).

    sigma: 1 / periodo di incubazione
    gamma: tasso di guarigione (costante o funzione del tempo)
    mu:    tasso di mortalità degli infetti (costante o funzione del tempo)
    """

    sigma: float
    gamma: Union[float, Callable[[float], float]]
    mu: Union[float, Callable[[float], float]]


def _at(rate, t: float) -> float:
    return rate(t) if callable(rate) else rate


def seird_rhs(t: float, y: np.ndarray, beta: Callable[[float], float],
              p: SEIRDParams) -> np.ndarray:
    S, E, I, R, D = y
    N = S + E + I + R  # i decessi escono dalla popolazione attiva
    infections = beta(t) * S * I / N
    gamma, mu = _at(p.gamma, t), _at(p.mu, t)
    return np.array([
        -infections,
        infections - p.sigma * E,
        p.sigma * E - (gamma + mu) * I,
        gamma * I,
        mu * I,
    ])


def simulate(beta: Callable[[float], float], params: SEIRDParams,
             y0: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """Integra il SEIRD e restituisce un array (len(t_eval), 5) con S, E, I, R, D."""
    y0 = np.asarray(y0, dtype=float)
    sol = solve_ivp(
        seird_rhs,
        (t_eval[0], t_eval[-1]),
        y0,
        t_eval=t_eval,
        args=(beta, params),
        method="LSODA",
        rtol=1e-6,
        atol=1e-8,
    )
    if not sol.success:
        raise RuntimeError(f"Integrazione fallita: {sol.message}")
    return sol.y.T


# --- Integratore veloce a passo fisso (per il GP) -------------------------------------------

SUBSTEPS = 4  # passi per giorno: h = 6 ore


def _on_grid(rate, grid: np.ndarray, n_days: int) -> np.ndarray:
    """Valori di un tasso sulla griglia fine. `rate` può essere:
    - un numero (costante);
    - un vettore di valori giornalieri (lunghezza n_days), interpolato linearmente come in
      `fitting._daily`;
    - una funzione del tempo in giorni (chiamata una volta per punto della griglia)."""
    if callable(rate):
        return np.fromiter((rate(t) for t in grid), float, len(grid))
    rate = np.asarray(rate, dtype=float)
    if rate.ndim == 0:
        return np.full(len(grid), float(rate))
    if len(rate) != n_days:
        raise ValueError(f"Servono {n_days} valori giornalieri, ricevuti {len(rate)}")
    return np.interp(grid, np.arange(n_days, dtype=float), rate)


def simulate_fixed(beta, params: SEIRDParams, y0, n_days: int,
                   substeps: int = SUBSTEPS) -> np.ndarray:
    """Come `simulate` sui giorni 0..n_days-1, ma con Runge-Kutta del 4° ordine a passo fisso.

    Molto più veloce di LSODA quando γ e μ variano ogni giorno: niente passi adattivi e i tassi
    si calcolano una volta sola sulla griglia, non a ogni chiamata del modello.
    `beta`, `params.gamma`, `params.mu`: numero, vettore giornaliero o funzione del tempo.
    Le discontinuità di β (es. baseline a tratti) cadono su giorni interi, che sono nodi
    della griglia: il passo fisso le tratta senza errori aggiuntivi.
    Equivalenza con `simulate` verificata in tests/test_model.py.
    """
    h = 1.0 / substeps
    n_steps = (n_days - 1) * substeps
    t0 = np.arange(n_steps) * h
    # I tassi si valutano *dentro* ogni passo: inizio un soffio dopo t, fine un soffio prima
    # di t + h. Così un salto di β su un giorno intero (baseline a tratti) non "sporca"
    # l'ultimo sotto-passo prima del salto. Per tassi continui non cambia nulla.
    eps = 1e-9
    stages = {"start": t0 + eps, "mid": t0 + h / 2, "end": t0 + h - eps}
    rates = {}
    for name, rate in [("b", beta), ("g", params.gamma), ("m", params.mu)]:
        for st, grid in stages.items():
            rates[name, st] = _on_grid(rate, grid, n_days).tolist()
    s = float(params.sigma)

    def rhs(S, E, I, R, j, st):
        b, g, m = rates["b", st][j], rates["g", st][j], rates["m", st][j]
        inf = b * S * I / (S + E + I + R)
        return -inf, inf - s * E, s * E - (g + m) * I, g * I, m * I

    S, E, I, R, D = (float(v) for v in y0)
    res = np.empty((n_days, 5))
    res[0] = S, E, I, R, D
    for j in range(n_steps):
        a1 = rhs(S, E, I, R, j, "start")
        a2 = rhs(S + h / 2 * a1[0], E + h / 2 * a1[1], I + h / 2 * a1[2], R + h / 2 * a1[3], j, "mid")
        a3 = rhs(S + h / 2 * a2[0], E + h / 2 * a2[1], I + h / 2 * a2[2], R + h / 2 * a2[3], j, "mid")
        a4 = rhs(S + h * a3[0], E + h * a3[1], I + h * a3[2], R + h * a3[3], j, "end")
        S += h / 6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0])
        E += h / 6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1])
        I += h / 6 * (a1[2] + 2 * a2[2] + 2 * a3[2] + a4[2])
        R += h / 6 * (a1[3] + 2 * a2[3] + 2 * a3[3] + a4[3])
        D += h / 6 * (a1[4] + 2 * a2[4] + 2 * a3[4] + a4[4])
        if (j + 1) % substeps == 0:
            res[(j + 1) // substeps] = S, E, I, R, D
    if not np.all(np.isfinite(res)):
        raise RuntimeError("Integrazione fallita: valori non finiti")
    return res