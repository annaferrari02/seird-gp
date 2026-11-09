import numpy as np

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
