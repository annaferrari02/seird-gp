import numpy as np
import pytest

from seird_gp import SEIRDParams, simulate
from seird_gp.data import TRENTO_POPULATION

PARAMS = SEIRDParams(sigma=1 / 5, gamma=1 / 10, mu=0.002)
Y0 = np.array([TRENTO_POPULATION - 70.0, 50.0, 20.0, 0.0, 0.0])
T = np.arange(0, 120, 1.0)


def test_population_is_conserved():
    y = simulate(lambda t: 0.3, PARAMS, Y0, T)
    assert np.allclose(y.sum(axis=1), Y0.sum(), rtol=1e-6)


def test_zero_beta_means_no_new_exposures():
    y = simulate(lambda t: 0.0, PARAMS, Y0, T)
    assert np.allclose(y[:, 0], Y0[0])


def test_states_stay_non_negative():
    y = simulate(lambda t: 0.5 if t < 40 else 0.05, PARAMS, Y0, T)
    assert (y > -1e-6).all()


def test_time_varying_rates_match_constants():
    # γ e μ passati come funzioni costanti danno lo stesso risultato dei numeri
    fn = SEIRDParams(sigma=1 / 5, gamma=lambda t: 1 / 10, mu=lambda t: 0.002)
    assert np.allclose(simulate(lambda t: 0.3, fn, Y0, T), simulate(lambda t: 0.3, PARAMS, Y0, T))


def test_higher_gamma_later_shortens_infection():
    late = SEIRDParams(sigma=1 / 5, gamma=lambda t: 1 / 10 if t < 60 else 1 / 3, mu=0.002)
    y_const = simulate(lambda t: 0.3, PARAMS, Y0, T)
    y_late = simulate(lambda t: 0.3, late, Y0, T)
    assert np.allclose(y_const[:60], y_late[:60], rtol=1e-4)
    assert y_late[-1, 2] < y_const[-1, 2]


# --- Integratore veloce a passo fisso: deve dare gli stessi risultati di LSODA ----------

from seird_gp.model import simulate_fixed  # noqa: E402

RNG = np.random.default_rng(0)
N_DAYS = len(T)
# γ(t) e μ(t) "rumorosi" giorno per giorno, come quelli letti dai dati
GAMMA_DAILY = np.clip(1 / 12 + 0.02 * RNG.standard_normal(N_DAYS), 0.02, None)
MU_DAILY = np.clip(0.002 + 0.001 * RNG.standard_normal(N_DAYS), 0.0, None)
BETAS = {
    "costante": lambda t: 0.3,
    "a gradino": lambda t: 0.45 if t < 40 else 0.12,     # salto su un giorno intero
    "oscillante": lambda t: 0.25 * (1 + 0.5 * np.sin(t / 7)),
}


def _interp(values):
    x = np.arange(len(values), dtype=float)
    return lambda t: float(np.interp(t, x, values))


def _max_rel_diff(a, b):
    """Differenza massima per compartimento, relativa al suo massimo."""
    scale = np.maximum(np.abs(a).max(axis=0), 1.0)
    return float((np.abs(a - b) / scale).max())


@pytest.mark.parametrize("name", list(BETAS))
def test_fixed_step_matches_lsoda_constant_rates(name):
    ref = simulate(BETAS[name], PARAMS, Y0, T)
    fast = simulate_fixed(BETAS[name], PARAMS, Y0, N_DAYS)
    assert _max_rel_diff(ref, fast) < 1e-4


@pytest.mark.parametrize("name", list(BETAS))
def test_fixed_step_matches_lsoda_daily_rates(name):
    # γ e μ come vettori giornalieri (rk4) o come funzioni interpolate (LSODA): stessa cosa
    ref = simulate(BETAS[name], SEIRDParams(1 / 5, _interp(GAMMA_DAILY), _interp(MU_DAILY)), Y0, T)
    fast = simulate_fixed(BETAS[name], SEIRDParams(1 / 5, GAMMA_DAILY, MU_DAILY), Y0, N_DAYS)
    assert _max_rel_diff(ref, fast) < 1e-4


def test_fixed_step_accepts_daily_beta_vector():
    beta_daily = np.array([BETAS["oscillante"](t) for t in T])
    a = simulate_fixed(beta_daily, PARAMS, Y0, N_DAYS)
    b = simulate(_interp(beta_daily), PARAMS, Y0, T)
    assert _max_rel_diff(a, b) < 1e-4


def test_fixed_step_conserves_population():
    y = simulate_fixed(BETAS["a gradino"], SEIRDParams(1 / 5, GAMMA_DAILY, MU_DAILY), Y0, N_DAYS)
    assert np.allclose(y.sum(axis=1), Y0.sum(), rtol=1e-9)


def test_fixed_step_rejects_wrong_length():
    with pytest.raises(ValueError):
        simulate_fixed(np.ones(N_DAYS - 1), PARAMS, Y0, N_DAYS)
