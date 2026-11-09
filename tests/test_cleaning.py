import pandas as pd
import pytest

from seird_gp.cleaning import Revision, apply_revisions, redistribute

IDX = pd.date_range("2020-03-01", periods=6)


def test_redistribute_removes_jump_and_keeps_final_value():
    cum = pd.Series([10, 20, 30, 40, 30, 32], index=IDX)  # -10 il 5° giorno
    rev = Revision("2020-03-05", "x", -10, "2020-03-01", "2020-03-04", "")
    out = redistribute(cum, rev)
    assert out.iloc[-1] == pytest.approx(cum.iloc[-1])
    assert out.diff().dropna().ge(0).all()  # niente più cali
    assert out.loc["2020-03-04"] == pytest.approx(30)  # 40 * (1 - 10/40)


def test_redistribute_rejects_revision_larger_than_window():
    cum = pd.Series([1, 2, 3, 4, -100, -100], index=IDX)
    rev = Revision("2020-03-05", "x", -104, "2020-03-01", "2020-03-04", "")
    with pytest.raises(ValueError):
        redistribute(cum, rev)


def test_apply_revisions_recomputes_positives():
    df = pd.DataFrame({
        "totale_positivi": [0] * 6,
        "dimessi_guariti": [0, 1, 2, 3, 4, 5],
        "deceduti": [1, 2, 3, 4, 2, 2],
        "totale_casi": [10, 20, 30, 40, 50, 60],
    }, index=IDX)
    rev = Revision("2020-03-05", "deceduti", -2, "2020-03-01", "2020-03-04", "")
    out = apply_revisions(df, [rev])
    assert (out.totale_positivi == out.totale_casi - out.dimessi_guariti - out.deceduti).all()
    assert out.deceduti.diff().dropna().ge(0).all()
