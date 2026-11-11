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