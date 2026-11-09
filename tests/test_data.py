import pandas as pd

from seird_gp.data import COLUMNS, MOBILITY_COLUMNS, load_mobility, load_trento

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


def test_load_mobility_keeps_trento_only(tmp_path):
    cols = ",".join(c + "_percent_change_from_baseline" for c in MOBILITY_COLUMNS)
    header = f"country_region_code,sub_region_2,iso_3166_2_code,date,{cols}\n"
    (tmp_path / "2020_IT_Region_Mobility_Report.csv").write_text(
        header
        + "IT,Autonomous Province of Trento,IT-TN,2020-03-02,-50,-10,,-60,-40,20\n"
        + "IT,Autonomous Province of Bolzano,IT-BZ,2020-03-02,-1,-1,-1,-1,-1,1\n"
    )
    (tmp_path / "2021_IT_Region_Mobility_Report.csv").write_text(
        header + "IT,Autonomous Province of Trento,IT-TN,2021-01-01,-30,-5,-20,-35,-25,10\n"
    )
    df = load_mobility(tmp_path)
    assert list(df.columns) == MOBILITY_COLUMNS
    assert len(df) == 2
    assert df.loc["2020-03-02", "workplaces"] == -40
    assert pd.isna(df.loc["2020-03-02", "parks"])  # NaN non imputati
