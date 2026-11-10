import numpy as np
import pandas as pd
import pytest

from seird_gp.fitting import (
    TRENTO_POPULATION, data_scale, fit_baseline, initial_state, normalized_fitness,
    pianesi_pojer_fitness, simulate_window,
)

TRUE = (0.35, 1 / 5, 1 / 20, 0.004, 300.0)  # beta, sigma, gamma, mu, E0


def synthetic_observations(days=60):
    idx = pd.date_range("2020-03-01", periods=days)
    seed = pd.DataFrame({"totale_positivi": [100.0], "dimessi_guariti": [0.0],
                         "deceduti": [0.0]}, index=idx[:1])
    sim = simulate_window(TRUE, seed.reindex(idx).ffill())
    return pd.DataFrame({"totale_positivi": sim["I"], "dimessi_guariti": sim["R"],
                         "deceduti": sim["D"]}, index=idx)


def test_initial_state_sums_to_population():
    obs = synthetic_observations(5)
    assert initial_state(obs, E0=123).sum() == pytest.approx(TRENTO_POPULATION)


def test_fit_recovers_known_parameters():
    obs = synthetic_observations()
    res = fit_baseline(obs, obs.index[0], obs.index[-1], train_frac=0.7, maxiter=150)
    assert res.fitness_train < 5
    assert res.params["beta"] == pytest.approx(TRUE[0], rel=0.1)
    # σ è il parametro meno determinato anche su dati sintetici senza rumore:
    # con 42 giorni di training si compensa con β ed E0 (σ ≈ 0.17 invece di 0.20)
    assert res.params["sigma"] == pytest.approx(TRUE[1], rel=0.2)
    assert res.params["gamma"] == pytest.approx(TRUE[2], rel=0.1)
    assert res.params["mu"] == pytest.approx(TRUE[3], rel=0.2)
    assert res.extra["at_bound"] == []
    assert set(res.rmsd.index) == {"train", "test"}
    # previsione fuori dal training: errore < 2% del picco degli infetti
    assert res.rmsd.loc["test", "I"] < 0.02 * obs["totale_positivi"].max()


def test_fixed_parameter_is_respected():
    obs = synthetic_observations(30)
    res = fit_baseline(obs, obs.index[0], obs.index[-1], fixed={"sigma": 0.2}, maxiter=30)
    assert res.params["sigma"] == pytest.approx(0.2)
    assert "sigma" not in res.extra["at_bound"]


def _sim_from_obs(obs, k_d=1.0, k_i=1.0):
    return pd.DataFrame({"D": obs["deceduti"] * k_d, "I": obs["totale_positivi"] * k_i})


def test_normalized_fitness_is_zero_on_perfect_fit():
    obs = synthetic_observations(30)
    assert normalized_fitness(_sim_from_obs(obs), obs) == pytest.approx(0.0)


def test_normalized_fitness_is_scale_invariant():
    # stesso errore relativo su serie con scale diverse -> stessa fitness
    obs = synthetic_observations(30)
    sim = _sim_from_obs(obs, 1.1, 1.1)
    big = obs * 1000
    assert normalized_fitness(sim, obs) == pytest.approx(normalized_fitness(sim * 1000, big))


def test_normalized_fitness_uses_given_scale():
    obs = synthetic_observations(30)
    sim = _sim_from_obs(obs, 1.1, 1.0)
    scale = {k: 2 * v for k, v in data_scale(obs).items()}
    assert normalized_fitness(sim, obs, scale=scale) == pytest.approx(
        normalized_fitness(sim, obs) / 2)


def test_fit_with_literature_parameters_and_normalized_fitness():
    # σ e γ fissati ai valori veri: restano da stimare β, μ, E0
    obs = synthetic_observations()
    res = fit_baseline(obs, obs.index[0], obs.index[-1], train_frac=0.7, maxiter=150,
                       fixed={"sigma": TRUE[1], "gamma": TRUE[2]},
                       fitness_fn=normalized_fitness)
    assert res.params["sigma"] == pytest.approx(TRUE[1])
    assert res.params["beta"] == pytest.approx(TRUE[0], rel=0.05)
    assert res.params["mu"] == pytest.approx(TRUE[3], rel=0.1)
    assert "fitness_norm" in res.rmsd.columns


def test_pianesi_pojer_fitness_uses_deaths_and_recovered():
    obs = synthetic_observations(30)
    sim = pd.DataFrame({"D": obs["deceduti"] + 10, "R": obs["dimessi_guariti"] + 100,
                        "I": obs["totale_positivi"] + 1e6})  # I non deve contare
    assert pianesi_pojer_fitness(sim, obs) == pytest.approx(0.9 * 10 + 0.1 * 100)
