"""Correzione delle revisioni retroattive nei dati DPC della P.A. Trento.

Quando la Provincia ricalcola un dato, la differenza compare tutta in un giorno solo
(es. -61 decessi il 24/06/2020). Qui la togliamo da quel giorno e la ridistribuiamo sul
periodo a cui si riferisce, in proporzione agli incrementi giornalieri osservati in quel
periodo. Il valore cumulativo finale non cambia.
"""

from dataclasses import dataclass

import pandas as pd

CUMULATIVE = ["dimessi_guariti", "deceduti", "totale_casi"]


@dataclass(frozen=True)
class Revision:
    date: str      # giorno in cui la revisione compare nei dati
    column: str    # colonna cumulativa interessata
    amount: float  # entità della revisione (negativa = rimozione)
    start: str     # inizio del periodo su cui ridistribuire
    end: str       # fine del periodo (incluso)
    note: str


KNOWN_REVISIONS = [
    # 22/05/2020: ~531 persone passano da guariti a positivi in un giorno.
    # Nessuna nota ufficiale trovata: ipotesi = guariti conteggiati in anticipo
    # nelle settimane precedenti e poi riallineati.
    Revision("2020-05-22", "dimessi_guariti", -531, "2020-02-24", "2020-05-21",
             "Riclassificazione guariti -> positivi (causa non documentata)"),
    # 24/06/2020: allineamento dei flussi Ministero/ISS (Protezione Civile).
    Revision("2020-06-24", "deceduti", -61, "2020-02-24", "2020-06-23",
             "Ricalcolo decessi per allineamento flussi"),
    Revision("2020-06-24", "dimessi_guariti", 447, "2020-02-24", "2020-06-23",
             "Ricalcolo guariti per allineamento flussi"),
    Revision("2020-06-24", "totale_casi", 387, "2020-02-24", "2020-06-23",
             "Ricalcolo casi per allineamento flussi"),
    # 24/01/2021: nota DPC, 30 decessi in due RSA di nov-dic 2020 non conteggiati.
    Revision("2021-01-24", "deceduti", 30, "2020-11-01", "2020-12-31",
             "30 decessi RSA di nov-dic 2020 conteggiati in ritardo"),
]


def to_daily(cum: pd.Series) -> pd.Series:
    daily = cum.astype(float).diff()
    daily.iloc[0] = cum.iloc[0]
    return daily


def redistribute(cum: pd.Series, rev: Revision) -> pd.Series:
    """Sposta `rev.amount` dal giorno `rev.date` al periodo [start, end]."""
    daily = to_daily(cum)
    window = daily.loc[rev.start:rev.end]
    total = window.sum()
    if total <= 0:
        raise ValueError(f"{rev.column}: nessun incremento nel periodo {rev.start}-{rev.end}")
    factor = 1 + rev.amount / total
    if factor < 0:
        raise ValueError(f"{rev.column}: revisione {rev.amount} più grande del periodo ({total})")
    daily.loc[rev.date] -= rev.amount
    daily.loc[rev.start:rev.end] = window * factor
    return daily.cumsum()


def apply_revisions(df: pd.DataFrame, revisions=KNOWN_REVISIONS) -> pd.DataFrame:
    """Applica le revisioni e ricalcola totale_positivi = casi - guariti - deceduti."""
    out = df.copy()
    for rev in revisions:
        if rev.date in out.index:
            out[rev.column] = redistribute(out[rev.column], rev)
    out["totale_positivi"] = out["totale_casi"] - out["dimessi_guariti"] - out["deceduti"]
    return out
