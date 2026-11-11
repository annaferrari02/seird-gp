import numpy as np
import pandas as pd
import pytest

from seird_gp.beta_models import (
    constant_beta, fit_beta, linear_mobility_beta, load_restrictions, piecewise_beta,
)
from seird_gp.fitting import simulate_window

SIGMA, GAMMA, MU, E0 = 1 / 5, 1 / 12, 0.003, 300.0
IDX = pd.date_range("2020-10-01", periods=90)


def observations(beta_fn):
    seed = pd.DataFrame({"totale_positivi": [100.0], "dimessi_guariti": [0.0],
                         "deceduti": [0.0]}, index=IDX[:1]).reindex(IDX).ffill()
    sim = simulate_window((np.nan, SIGMA, GAMMA, MU, E0), seed, beta_fn=beta_fn)
    return pd.DataFrame({"totale_positivi": sim["I"], "dimessi_guariti": sim["R"],
                         "deceduti": sim["D"]}, index=IDX)


FIXED = {"sigma": SIGMA, "gamma": GAMMA, "mu": MU}


def test_constant_model_recovers_beta():
    obs = observations(lambda t: 0.3)
    res = fit_beta(obs, IDX[0], IDX[-1], constant_beta(), train_frac=0.7, fixed=FIXED, maxiter=100)
    assert res.params["beta"] == pytest.approx(0.3, rel=0.03)


def test_piecewise_recovers_steps_and_ignores_future_breaks():
    breaks = [IDX[30], IDX[80]]  # il secondo cade nel test (training = primi 63 giorni)
    obs = observations(lambda t: 0.35 if t < 30 else 0.15)
    res = fit_beta(obs, IDX[0], IDX[-1], piecewise_beta(breaks), train_frac=0.7,
                   fixed=FIXED, maxiter=150)
    assert res.params["beta_0"] == pytest.approx(0.35, rel=0.05)
    assert res.params["beta_1"] == pytest.approx(0.15, rel=0.05)
    assert "beta_2" not in res.params           # periodo solo-test: non stimato
    assert res.extra["beta"].iloc[-1] == pytest.approx(res.params["beta_1"])


def test_piecewise_skips_periods_too_short_in_training():
    # training = primi 63 giorni (IDX[0]..IDX[62]); un periodo che inizia a IDX[58] ha 5 giorni
    model = piecewise_beta([IDX[30], IDX[58]], min_train_days=7)
    assert model.active_params(IDX, IDX[62]) == ("beta_0", "beta_1")
    f = model.build({"beta_0": 0.3, "beta_1": 0.2, "beta_2": np.nan}, IDX, IDX[62])
    assert f(80) == 0.2                          # il periodo corto resta al β precedente
    assert piecewise_beta([IDX[30], IDX[56]]).active_params(IDX, IDX[62]) == \
        ("beta_0", "beta_1", "beta_2")           # 7 giorni: si stima


def test_linear_mobility_recovers_coefficient():
    m = pd.DataFrame({"x": -0.5 * (np.arange(len(IDX)) > 40)}, index=IDX)  # mobilità a gradino
    true = 0.32 * (1 + 1.2 * m["x"].to_numpy())
    obs = observations(lambda t: float(np.interp(t, np.arange(len(IDX)), true)))
    res = fit_beta(obs, IDX[0], IDX[-1], linear_mobility_beta(m, ["x"]), train_frac=0.7,
                   fixed=FIXED, maxiter=150)
    assert res.params["beta0"] == pytest.approx(0.32, rel=0.05)
    assert res.params["c_x"] == pytest.approx(1.2, rel=0.1)


def test_linear_mobility_never_negative():
    m = pd.DataFrame({"x": np.full(len(IDX), -0.9)}, index=IDX)
    f = linear_mobility_beta(m, ["x"]).build({"beta0": 0.3, "c_x": 5.0}, IDX, IDX[-1])
    assert f(10) == 0.0


def test_restrictions_table_is_sorted_and_complete():
    r = load_restrictions()
    assert r["inizio"].is_monotonic_increasing
    assert not r[["inizio", "livello", "provvedimento", "fonte"]].isna().any().any()
