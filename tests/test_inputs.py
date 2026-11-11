import numpy as np
import pandas as pd
import pytest

from seird_gp.inputs import build_gp_inputs, check_coverage, smooth_past

DAYS = pd.date_range("2020-02-15", periods=60, freq="D")


def _mobility(seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"retail_and_recreation": rng.normal(-20, 10, len(DAYS)),
                         "workplaces": rng.normal(-10, 5, len(DAYS))}, index=DAYS)


def test_smoothing_uses_only_past():
    mob = _mobility()
    changed = mob.copy()
    changed.iloc[30:] += 50  # cambia solo il futuro rispetto al giorno 29
    pd.testing.assert_frame_equal(smooth_past(mob).iloc[:30], smooth_past(changed).iloc[:30])


def test_smoothing_is_7_day_mean():
    mob = _mobility()
    assert smooth_past(mob).iloc[20, 0] == pytest.approx(mob.iloc[14:21, 0].mean())


def test_normalization_and_lags():
    mob = _mobility()
    inp = build_gp_inputs(mob)
    smoothed = smooth_past(mob) / 100
    assert inp["retail_lag0"].iloc[30] == pytest.approx(smoothed["retail_and_recreation"].iloc[30])
    assert inp["work_lag7"].iloc[30] == pytest.approx(smoothed["workplaces"].iloc[23])
    assert inp["retail_lag14"].iloc[:14].isna().all()
    assert list(inp.columns) == [f"{n}_lag{k}" for n in ("retail", "work") for k in (0, 3, 7, 10, 14)]


def test_coverage_passes_and_fails():
    inp = build_gp_inputs(_mobility())
    ok = check_coverage(inp, {"dentro": ("2020-03-01", "2020-04-01")})  # dal 15/02: 14 giorni prima ci sono
    assert ok.to_numpy().sum() == 0
    with pytest.raises(ValueError):
        check_coverage(inp, {"troppo presto": ("2020-02-20", "2020-03-10")})


def test_missing_index_raises():
    with pytest.raises(KeyError):
        build_gp_inputs(_mobility()[["workplaces"]])