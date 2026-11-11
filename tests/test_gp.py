import numpy as np
import pandas as pd
import pytest

from seird_gp.gp import (
    BETA_MAX, BETA_MIN, CONFIGS, GPConfig, MOBILITY_TERMINALS, Problem, knee, protected_div, run_gp,
)
from seird_gp.model import SEIRDParams, simulate_fixed

IDX = pd.date_range("2020-09-01", periods=120)
N = 542_166.0
SIGMA, E0 = 1 / 5.2, 150.0
BETA_REF = float(np.exp(-2.0))  # β della 'baseline costante' sintetica


def synthetic_problem(log_beta):
    """Epidemia simulata con β = exp(log_beta(X)) nota, γ e μ costanti."""
    t = np.arange(len(IDX))
    work = -0.15 - 0.35 / (1 + np.exp(-(t - 60) / 6))           # mobilità che cala a metà
    X = pd.DataFrame({c: 0.0 for c in MOBILITY_TERMINALS}, index=IDX)
    X["work_lag0"] = work
    X["retail_lag0"] = 0.3 * np.sin(t / 9)                        # distrattore
    X["t"] = t / 30
    rates = pd.DataFrame({"gamma": np.full(len(IDX), 1 / 12), "mu": np.full(len(IDX), 0.002)}, index=IDX)
    y0 = np.array([N - E0 - 100, E0, 100, 0, 0])
    y = simulate_fixed(np.exp(log_beta(X)), SEIRDParams(SIGMA, 1 / 12, 0.002), y0, len(IDX))
    obs = pd.DataFrame({"totale_positivi": y[:, 2], "dimessi_guariti": y[:, 3], "deceduti": y[:, 4]},
                       index=IDX)
    return Problem(obs, 80, X, SIGMA, E0, rates, beta_ref=BETA_REF)


def test_protected_div():
    assert protected_div(1.0, 0.0) == 1.0
    assert np.allclose(protected_div(np.array([2.0, 3.0]), np.array([0.0, 2.0])), [1.0, 1.5])


def test_true_formula_has_zero_error():
    problem = synthetic_problem(lambda X: -2.0 + 1.5 * X["work_lag0"])
    func = lambda work_lag0: 1.5 * work_lag0  # noqa: E731  (g = scarto da BETA_REF = e^-2)
    assert problem.train_error(func, ["work_lag0"]) < 1e-6


def test_beta_is_clipped():
    problem = synthetic_problem(lambda X: -2.0 + 0 * X["t"])
    b = problem.beta(lambda t: 100 * np.sin(t * 30), ["t"])
    assert b.min() >= BETA_MIN * (1 - 1e-12) and b.max() <= BETA_MAX * (1 + 1e-12)


def test_bad_formula_does_not_crash():
    problem = synthetic_problem(lambda X: -2.0 + 0 * X["t"])
    assert problem.train_error(lambda t: t * np.inf, ["t"]) >= 1e5


def test_gp_recovers_mobility_effect():
    # β = β_ref · exp(1.5 · work): il GP con la mobilità deve ritrovarla (seed fisso, ~6 s).
    # Su 8 seed la ritrova esattamente in 6; gli altri si fermano su "work + costante".
    problem = synthetic_problem(lambda X: -2.0 + 1.5 * X["work_lag0"])
    front, _ = run_gp(problem, CONFIGS["mobilita"], GPConfig(pop_size=150, n_offspring=150, n_gen=25),
                      seed=1)
    best = front.loc[front["fitness_norm train"].idxmin()]
    assert best["fitness_norm train"] < 0.01
    assert best["terminali usati"] == ["work_lag0"]
    assert front["nodi"].is_monotonic_increasing
    assert knee(front) in front.index
