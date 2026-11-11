"""Input di mobilità per il GP: indici scelti, media mobile all'indietro, normalizzazione, ritardi.

Scelte (notebook 03):
- indici: retail_and_recreation e workplaces, la coppia meno correlata fra le categorie
  legate ai contatti (|r| = 0.60 sull'intero periodo, serie lisciate);
- media mobile a 7 giorni solo all'indietro: il valore di un giorno usa solo dati fino a quel giorno;
- normalizzazione: variazione % / 100 (-40% -> -0.4), confrontabile fra finestre;
- ritardi 0, 3, 7, 10, 14 giorni: il GP sceglie quali usare. In teoria il ritardo giusto è 0
  (l'incubazione è già nel compartimento E); i ritardi lunghi servono come diagnostica di un
  ritardo di notifica non modellato (criterio nel notebook 03).

I ritardi vanno costruiti dalla serie di mobilità originale (dal 15/02/2020), non dal file
unito con i dati COVID (dal 24/02/2020), altrimenti nella finestra 2020 mancano i ritardi lunghi.
"""

import pandas as pd

GP_INDICES = {"retail_and_recreation": "retail", "workplaces": "work"}
LAGS = (0, 3, 7, 10, 14)
SMOOTH_DAYS = 7


def smooth_past(mobility: pd.DataFrame, days: int = SMOOTH_DAYS) -> pd.DataFrame:
    """Media mobile all'indietro. Ai primi giorni della serie la media usa i giorni disponibili
    (meno di `days`): sono giorni pre-pandemia, vicini alla baseline."""
    return mobility.rolling(days, min_periods=1).mean()


def build_gp_inputs(mobility: pd.DataFrame, indices: dict = GP_INDICES, lags=LAGS,
                    days: int = SMOOTH_DAYS) -> pd.DataFrame:
    """Colonne `<nome>_lag<k>` = indice lisciato e normalizzato, k giorni prima.

    `mobility` = serie originale da `load_mobility()`, con le colonne di `indices`.
    """
    missing = [c for c in indices if c not in mobility]
    if missing:
        raise KeyError(f"Indici di mobilità mancanti: {missing}")
    base = smooth_past(mobility[list(indices)], days) / 100
    base = base.rename(columns=indices)
    cols = {f"{name}_lag{k}": base[name].shift(k) for name in indices.values() for k in lags}
    return pd.DataFrame(cols, index=mobility.index)


def check_coverage(inputs: pd.DataFrame, windows: dict) -> pd.DataFrame:
    """Per ogni finestra, quanti giorni mancano a ogni colonna. Errore se ne manca qualcuno."""
    table = pd.DataFrame({
        name: inputs.reindex(pd.date_range(start, end, freq="D")).isna().sum()
        for name, (start, end) in windows.items()
    })
    if table.to_numpy().sum() > 0:
        bad = table[table.sum(axis=1) > 0]
        raise ValueError(f"Input mancanti nelle finestre:\n{bad}")
    return table