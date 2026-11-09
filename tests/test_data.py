import pandas as pd

from seird_gp.data import COLUMNS, load_trento

CSV = """data,stato,codice_regione,denominazione_regione,totale_positivi,dimessi_guariti,deceduti,totale_casi
2020-03-02T18:00:00,ITA,22,P.A. Trento,5,0,0,5
2020-03-01T18:00:00,ITA,22,P.A. Trento,2,0,0,2
2020-03-01T18:00:00,ITA,21,P.A. Bolzano,1,0,0,1
"""


def test_load_trento_filters_and_sorts(tmp_path):
    path = tmp_path / "dpc.csv"
    path.write_text(CSV)
    df = load_trento(path)
    assert list(df.columns) == COLUMNS
    assert len(df) == 2  # Bolzano escluso
    assert df.index.is_monotonic_increasing
    assert df.index[0] == pd.Timestamp("2020-03-01")
    assert df["totale_positivi"].tolist() == [2, 5]
