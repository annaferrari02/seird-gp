import numpy as np
import pandas as pd
import pytest

from seird_gp.gp import MOBILITY_TERMINALS, make_problem
from seird_gp.synthetic import SCENARIOS, generate, oracle, true_beta

START, END = "2020-09-01", "2020-12-29"
IDX = pd.date_range("2020-08-01", END)
WINDOW = pd.date_range(START, END)
N_TRAIN = int(round(len(WINDOW) * 2 / 3))


@pytest.fixture(scope="module")
def real():
    """Serie 'vera' finta ma coerente: γ e μ costanti, positivi in crescita lenta."""
    t = np.arange(len(IDX))
    I = 100 * np.exp(t / 60)
    R = 4000 + np.cumsum(I / 15)
    D = 400 + np.cumsum(I * 0.002)
    return pd.DataFrame({"totale_positivi": I, "dimessi_guariti": R, "deceduti": D}, index=IDX)


@pytest.fixture(scope="module")
def X():
    t = np.arange(len(IDX))
    work = -0.1 - 0.4 / (1 + np.exp(-(t - 90) / 8))
    cols = {}
    for name, base in [("work", work), ("retail", 0.2 * np.sin(t / 11) - 0.2)]:
        for k in (0, 3, 7, 10, 14):
            cols[f"{name}_lag{k}"] = pd.Series(base, index=IDX).shift(k).bfill().to_numpy()
    return pd.DataFrame(cols, index=IDX)[MOBILITY_TERMINALS]


RESTR = pd.DataFrame({"inizio": pd.to_datetime(["2020-03-10", "2020-11-06"]), "livello": ["lockdown", "gialla"]})


def test_level_is_geometric_mean_on_training(X):
    for sc in SCENARIOS.values():
        b = true_beta(sc, X.reindex(WINDOW), WINDOW, N_TRAIN, 0.1, RESTR)
        assert np.exp(np.log(b[:N_TRAIN]).mean()) == pytest.approx(0.1)


def test_restriction_steps_only_inside_window(X):
    b = true_beta(SCENARIOS["C"], X.reindex(WINDOW), WINDOW, N_TRAIN, 0.1, RESTR)
    before, after = b[WINDOW < "2020-11-06"], b[WINDOW >= "2020-11-06"]
    assert np.ptp(before) == 0 and np.ptp(after) == 0       # il lockdown di marzo non conta
    assert after[0] / before[0] == pytest.approx(0.75)


def test_mobility_scenario_uses_lag7(X):
    b = true_beta(SCENARIOS["A"], X.reindex(WINDOW), WINDOW, N_TRAIN, 0.1)
    g = np.log(b)
    assert np.corrcoef(g, X.reindex(WINDOW)["work_lag7"])[0, 1] == pytest.approx(1.0)


def test_noise_free_oracle_is_exact_with_true_rates(real, X):
    b = true_beta(SCENARIOS["A"], X.reindex(WINDOW), WINDOW, N_TRAIN, 0.1)
    df, truth = generate(real, X.reindex(WINDOW), START, END, b, 1 / 4, noise=False)
    problem = make_problem(df, X, START, END, 1 / 4, 0.1)
    o = oracle(problem, truth)
    assert o["oracolo, γ μ veri train"] < 1e-3 and o["oracolo, γ μ veri test"] < 1e-3
    assert o["solo rumore train"] == pytest.approx(0) and o["oracolo, γ μ veri E0"] == pytest.approx(150, rel=0.02)


def test_noise_is_reproducible_and_cumulatives_increase(real, X):
    b = true_beta(SCENARIOS["D"], X.reindex(WINDOW), WINDOW, N_TRAIN, 0.1)
    df1, _ = generate(real, X.reindex(WINDOW), START, END, b, 1 / 4, seed=3)
    df2, _ = generate(real, X.reindex(WINDOW), START, END, b, 1 / 4, seed=3)
    pd.testing.assert_frame_equal(df1, df2)
    w = df1.loc[START:END]
    assert (w[["dimessi_guariti", "deceduti"]].diff().dropna() >= 0).all().all()
    # i giorni prima della finestra restano quelli veri (servono a empirical_rates)
    pd.testing.assert_frame_equal(df1.loc[:"2020-08-31"], real.loc[:"2020-08-31", list(df1.columns)])
