"""Download e preprocessing dei dati COVID-19 della P.A. Trento (fonte: Protezione Civile)."""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

DPC_URL = (
    "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/"
    "dati-regioni/dpc-covid19-ita-regioni.csv"
)
TRENTO_CODE = 22
TRENTO_POPULATION = 542_166  # ISTAT, 31/12/2020 (come in Pianesi e Pojer)

# Colonne tenute da Pianesi e Pojer.
# totale_positivi = positivi attuali (prevalenza); le altre sono cumulative.
COLUMNS = ["totale_positivi", "dimessi_guariti", "deceduti", "totale_casi"]

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"


def download_dpc(dest_dir: Path = RAW_DIR, overwrite: bool = False) -> Path:
    """Scarica il CSV regionale completo della Protezione Civile in data/raw/."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / "dpc-covid19-ita-regioni.csv"
    if overwrite or not path.exists():
        urlretrieve(DPC_URL, path)
    return path


def load_trento(path: Path) -> pd.DataFrame:
    """Filtra il CSV DPC sulla P.A. Trento.

    Restituisce un DataFrame indicizzato per giorno (DatetimeIndex) con le colonne in COLUMNS.
    """
    df = pd.read_csv(path)
    df = df[df["codice_regione"] == TRENTO_CODE]
    df["data"] = pd.to_datetime(df["data"]).dt.normalize()
    df = df.set_index("data").sort_index()[COLUMNS]
    if df.index.duplicated().any():
        raise ValueError("Date duplicate nei dati di Trento")
    return df


def build_trento_dataset(overwrite: bool = False) -> pd.DataFrame:
    """Scarica (se serve), filtra e salva data/processed/trento_covid.csv."""
    df = load_trento(download_dpc(overwrite=overwrite))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "trento_covid.csv")
    return df


# --- Mobilità (Google COVID-19 Community Mobility Reports) -----------------
# I CSV vanno scaricati a mano da
# https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip
# e messi in data/raw/ come 2020_IT_Region_Mobility_Report.csv, 2021_..., 2022_...

TRENTO_ISO = "IT-TN"
MOBILITY_SUFFIX = "_percent_change_from_baseline"
MOBILITY_COLUMNS = [
    "retail_and_recreation",
    "grocery_and_pharmacy",
    "parks",
    "transit_stations",
    "workplaces",
    "residential",
]


def load_mobility(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Legge i report Google italiani e tiene la P.A. Trento.

    Valori = variazione % rispetto alla baseline (mediana gen-feb 2020 per giorno della settimana).
    I valori mancanti restano NaN: l'eventuale imputazione si decide a valle.
    """
    files = sorted(raw_dir.glob("20*_IT_Region_Mobility_Report.csv"))
    if not files:
        raise FileNotFoundError(f"Nessun report di mobilità in {raw_dir}")
    df = pd.concat(pd.read_csv(f, low_memory=False) for f in files)
    df = df[df["iso_3166_2_code"] == TRENTO_ISO]
    df = df.rename(columns={c + MOBILITY_SUFFIX: c for c in MOBILITY_COLUMNS})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()[MOBILITY_COLUMNS]
    df.index.name = "data"
    if df.index.duplicated().any():
        raise ValueError("Date duplicate nei dati di mobilità")
    return df


def build_merged_dataset() -> pd.DataFrame:
    """Unisce COVID e mobilità sui giorni in comune; salva data/processed/trento_merged.csv."""
    covid = build_trento_dataset()
    merged = covid.join(load_mobility(), how="inner")
    merged.to_csv(PROCESSED_DIR / "trento_merged.csv")
    return merged


if __name__ == "__main__":
    out = build_trento_dataset()
    print(f"COVID: {len(out)} giorni, dal {out.index.min().date()} al {out.index.max().date()}")
    try:
        m = build_merged_dataset()
        print(f"Unito con mobilità: {len(m)} giorni, "
              f"dal {m.index.min().date()} al {m.index.max().date()}")
    except FileNotFoundError as e:
        print(f"Mobilità saltata: {e}")
