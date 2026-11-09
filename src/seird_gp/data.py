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


if __name__ == "__main__":
    out = build_trento_dataset()
    print(f"{len(out)} giorni, dal {out.index.min().date()} al {out.index.max().date()}")
