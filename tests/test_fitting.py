import numpy as np
import pandas as pd
import pytest

from seird_gp.fitting import (
    TRENTO_POPULATION, fit_baseline, initial_state, simulate_window,
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
    assert res.params["mu"] == pytest.approx(TRUE[3], rel=0.2)
    assert set(res.rmsd.index) == {"train", "test"}
    # previsione fuori dal training: errore < 2% del picco degli infetti
    assert res.rmsd.loc["test", "I"] < 0.02 * obs["totale_positivi"].max()


def test_fixed_parameter_is_respected():
    obs = synthetic_observations(30)
    res = fit_baseline(obs, obs.index[0], obs.index[-1], fixed={"sigma": 0.2}, maxiter=30)
    assert res.params["sigma"] == pytest.approx(0.2)
    assert "sigma" not in res.extra["at_bound"]
